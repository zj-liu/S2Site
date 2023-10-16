import numpy as np
import sys
import os

import torch

from preprocessing import PDB_processing, PDBio, pipelines
from utilities import configuration as cfg
from utilities import wrappers, chimera, evaluate, dataloader
from networks import rcnn, s2site, unet
import time

import argparse


def write_predictions(csv_file, residue_ids, sequence, interface_prediction):
    L = len(residue_ids)
    columns = ['Model','Chain','Residue Index','Sequence']
    if interface_prediction.ndim == 1:
        columns.append('Binding site probability')
    else:
        columns += ['Output %s' %i for i in range(interface_prediction.shape[-1] )]

    with open(csv_file, 'w') as f:
        f.write(','.join(columns) + '\n' )
        for i in range(L):
            string = '%s,%s,%s,%s,' % (residue_ids[i][0],
                                       residue_ids[i][1],
                                       residue_ids[i][2],
                                       sequence[i])
            # import pdb; pdb.set_trace()
            if interface_prediction.ndim == 1:
                string += '%.3f'%interface_prediction[i]
                # print('write i:', i)
            else:
                string += ','.join(['%.3f'%value for value in interface_prediction[i]])
            f.write(string + '\n')
    return


def predict_interface_residues(
    config,
    query_pdbs, # '1a3x'
    pipeline,
    model,
    model_name,
    model_type,
    query_chain_ids=None,
    query_sequences=None,
    query_names=None,
    logfile=None,
    biounit=True,
    assembly=True,
    Lmin=1,
    return_all=False,
    output_predictions=True,
    aggregate_models=True,
    output_chimera='annotation',
    chimera_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    permissive=False,
    output_format='numpy',
    device='cpu'):
    
    read_s = time.time()
    
    predictions_folder = config.predicts
    if not os.path.isdir(predictions_folder):
        os.mkdir(predictions_folder)

    if query_pdbs is not None:
        try:  # Check whether chain_ids is a list of pdb/chains or a single pdb/chain
            assert len(query_pdbs[0]) > 1
        except:
            query_pdbs = [query_pdbs]
        print('Predicting binding sites from pdb structures with %s' % model_name, file=logfile)
        predict_from_pdb = True
        predict_from_sequence = False
        npdbs = len(query_pdbs)

        if query_chain_ids is None:
            query_chain_ids = ['all' for _ in query_pdbs]
        else:
            if not ( (query_chain_ids[0] in ['all','upper','lower'] ) | (isinstance(query_chain_ids[0], list)) ):
                query_chain_ids = [query_chain_ids]
    elif query_sequences is not None:
        try:  # Check whether sequences is a list of pdb/chains or a single pdb/chain
            assert len(query_sequences[0]) > 1
        except:
            query_sequences = [query_sequences]
        print('Predicting interface residues from sequences using %s' %
              model_name, file=logfile)
        predict_from_pdb = False
        predict_from_sequence = True
        nqueries = len(query_sequences)
    else:
        print('No input provided for interface prediction using %s' %
              model_name, file=logfile)
        return

    if query_names is None:
        if predict_from_pdb:
            query_names = []
            for i in range(npdbs):
                pdb = query_pdbs[i].split('/')[-1].split('.')[0]
                query_names.append(pdb)
        elif predict_from_sequence:
            sequence_lengths = [len(sequence) for sequence in query_sequences]
            first_aa = [sequence[:5] for sequence in query_sequences]
            query_names = ['seq_%s_start:%s_L:%s' % (
                i, first_aa[i], sequence_lengths[i]) for i in range(nqueries)]

    # Locate pdb files or download from pdb server.
    if predict_from_pdb:
        pdb_file_locations = []
        i = 0
        while i < npdbs:
            pdb_id = query_pdbs[i]
            location, chain = PDBio.getPDB(pdb_id, biounit=biounit, structures_folder=config.structures_folder, 
                                           download_noFound=config.download_noFound)
            if not os.path.exists(location):
                print('i=%s, file:%s not found' % (i, location), file=logfile)
                if permissive & (npdbs > 1):
                    del query_pdbs[i]
                    del query_chain_ids[i]
                    del query_names[i]
                    npdbs -= 1
                else:
                    return
            else:
                pdb_file_locations.append(location)
                i += 1

        # Parse pdb files.
        query_chain_objs = []
        query_chain_names = []
        query_chain_id_is_alls = [query_chain_id == 'all' for query_chain_id in query_chain_ids]

        i = 0
        while i < npdbs:
            try:
                _, chain_objs = PDBio.load_chains(
                    chain_ids= query_chain_ids[i], file=pdb_file_locations[i], structures_folder=config.structures_folder)

                if query_chain_ids[i] == 'all':
                    query_chain_ids[i] = [(chain_obj.get_full_id()[1], chain_obj.get_full_id()[2])
                                             for chain_obj in chain_objs]
                elif query_chain_ids[i] == 'upper':
                    query_chain_ids[i] = [(chain_obj.get_full_id()[1], chain_obj.get_full_id()[2])
                                             for chain_obj in chain_objs if (chain_obj.get_full_id()[2].isupper() | (chain_obj.get_full_id()[2] == ' ') )]
                elif query_chain_ids[i] == 'lower':
                    query_chain_ids[i] = [(chain_obj.get_full_id()[1], chain_obj.get_full_id()[2])
                                             for chain_obj in chain_objs if chain_obj.get_full_id()[2].islower()]

                query_chain_objs.append(chain_objs)

                query_chain_names.append([query_names[i] + '_%s_%s' %
                                      query_chain_id for query_chain_id in query_chain_ids[i]])

                i += 1
            except:
                print('Failed to parse i=%s,%s, %s' %
                      (i, query_names[i], pdb_file_locations[i]), file=logfile, download_noFound=config.download_noFound)
                if permissive & (npdbs > 1):
                    del query_pdbs[i]
                    del query_chain_ids[i]
                    del query_names[i]
                    npdbs -= 1
                else:
                    return

        query_sequences = [[PDB_processing.process_chain(chain_obj)[0]
                            for chain_obj in chain_objs] for chain_objs in query_chain_objs]

        if Lmin > 0:
            for i in range(npdbs):
                j = 0
                nsequences = len(query_sequences[i])
                while j < nsequences:
                    sequence = query_sequences[i][j]
                    if len(sequence) < 2:
                        print('Chain %s %s from PDB %s is too short (L=%s), discarding.' % (
                        query_chain_ids[i][j][0], query_chain_ids[i][j][1], query_pdbs[i], len(sequence)), file=logfile)
                        del query_sequences[i][j]
                        del query_chain_ids[i][j]
                        del query_chain_objs[i][j]
                        del query_chain_names[i][j]
                        nsequences -= 1
                    else:
                        j += 1

        i = 0
        while i < npdbs:
            if not len(query_sequences[i]) > 0:
                print('PDB %s has no chains remaining!' % (query_pdbs[i]))
                if permissive & (npdbs > 1):
                    del query_pdbs[i]
                    del query_sequences[i]
                    del query_chain_ids[i]
                    del query_chain_objs[i]
                    del query_names[i]
                    del query_chain_names[i]
                    npdbs -= 1
                else:
                    return
            else:
                i += 1

        nqueries = npdbs
    else:
        query_chain_names = query_names
        query_chain_objs = [None for _ in query_chain_names]
        if query_chain_ids is None:
            query_chain_ids = [('', '') for _ in query_chain_names]
        nqueries = len(query_names)

    print('List of inputs:', file=logfile)
    for i in range(nqueries):
        print(query_chain_names[i], file=logfile)
       
    sequence_lengths = [[len(sequence) for sequence in sequences]
                           for sequences in query_sequences]
    if assembly:
        assembly_lengths = [sum(sequence_length)
                              for sequence_length in sequence_lengths]
        Lmax = max(assembly_lengths)
    else:
        Lmax = max([max(sequence_length)
                 for sequence_length in sequence_lengths])
    Lmax = max(Lmax, 32)
    
    query_residue_ids =[]
    query_sequences=[''.join(sequences) for sequences in query_sequences]
    for i, chain_objs in enumerate(query_chain_objs):
        if chain_objs is not None:
            residue_ids =  PDB_processing.get_PDB_indices(chain_objs, return_chain=True, return_model=True)
        else:
            model_indices=[' ' for _ in query_sequences[i]]
            chain_indices=[' ' for _ in query_sequences[i]]
            residue_indices= ['%s'%i for i in range(1, len(query_sequences[i]) + 1) ]
            residue_ids = np.concatenate(
                np.array(model_indices)[:, np.newaxis],
                np.array(chain_indices)[:, np.newaxis],
                np.array(residue_indices)[:, np.newaxis],
                axis = 1
             )

        query_residue_ids.append( residue_ids)
    
    print('Loading model %s' % model_name, file=logfile)
    if isinstance(model, list):
        multi_models = True
        ckpts = [torch.load(config.model_folder + model_) for model_ in model]
        model_wrappers = []
        model_objs = []
        model_obj = None
        for ckpt in ckpts:
            ckpt['config_setting'].reset_path()
            ckpt['config_setting'].get_setting().Lmax_aa = Lmax
            ckpt['config_setting'].get_setting().device = device
            if model_type == 'scannet':
                model, wrapper, info = s2site.initial_S2Site(config=ckpt['config_setting'].get_setting())
                model_wrappers.append(wrapper)
            elif model_type == 'rcnn':
                config_ = ckpt['config_setting'].get_setting()
                model = rcnn.RCNN(in_channels=config_.nfeatures_aa, hidden_channels=config_.nembedding_aa, 
                            num_rcnn_blocks=config_.num_layer_blocks, gru_layer=config_.num_gru_layers, 
                            gru_dropout=config_.gru_dropout, with_pool=config_.with_pool, dropout_rate=config_.dropout)
            else:
                config_ = ckpt['config_setting'].get_setting()
                model = unet.UNet(in_features=config_.nfeatures_aa, conv_features=config_.nembedding_aa, num_layers=config_.num_layer_blocks, dropout_rate=config_.dropout)

            model.load_state_dict(ckpt['model_state'])
            model_objs.append(model)
            
    else:
        multi_models = False
        ckpt = torch.load(config.model_folder + model)
        ckpt['config_setting'].reset_path()
        ckpt['config_setting'].get_setting().device = device
        ckpt['config_setting'].get_setting().Lmax_aa = Lmax
        if model_type == 'scannet':
            model_obj, model_wrapper, info = s2site.initial_S2Site(config=ckpt['config_setting'].get_setting())
        elif model_type == 'rcnn':
            config_ = ckpt['config_setting'].get_setting()
            model_obj = rcnn.RCNN(in_channels=config_.nfeatures_aa, hidden_channels=config_.nembedding_aa, 
                        num_rcnn_blocks=config_.num_layer_blocks, gru_layer=config_.num_gru_layers, 
                        gru_dropout=config_.gru_dropout, with_pool=config_.with_pool, dropout_rate=config_.dropout)
        else:
            config_ = ckpt['config_setting'].get_setting()
            model_obj = unet.UNet(in_features=config_.nfeatures_aa, conv_features=config_.nembedding_aa, num_layers=config_.num_layer_blocks, dropout_rate=config_.dropout)
            
        model_obj.load_state_dict(ckpt['model_state'])
        model_objs = None
        
    if hasattr(pipeline, 'Lmax'):
        pipeline.Lmax = Lmax
    if hasattr(pipeline, 'Lmax_aa'):
        pipeline.Lmax_aa = Lmax
    if hasattr(pipeline, 'Lmax_atom'):
        pipeline.Lmax_atom = 9* Lmax

    if hasattr(pipeline, 'padded'):
        padded = pipeline.padded
    else:
        padded = True

    if assembly:
        # list of 8 inputs
        inputs = wrappers.stack_list_of_arrays([pipeline.process_example(chain_obj=chain_obj, sequence=sequence)[0] 
                                                for chain_obj, sequence in zip(query_chain_objs, query_sequences)], padded=padded)
        if model_type != 'scannet':
            datas = dataloader.Baseline_Dataset(x=inputs, y=None, sample_weight=None, mode='val', num_layer_blocks=config_.num_layer_blocks, model_type=model_type)
        
        if multi_models:
            if aggregate_models:
                if model_type == 'scannet':
                    model_objs[0].reset_device(device)
                    query_predictions = evaluate.test_predicts(inputs=inputs, model=model_objs[0].reset_device(device), wrapper=model_wrappers[0], return_all=return_all)[0]
                    for i, model_obj in enumerate(model_objs[1:]):
                        torch.cuda.empty_cache()
                        model_obj.reset_device(device)
                        predictions = evaluate.test_predicts(inputs=inputs, model=model_obj, wrapper=model_wrappers[i], return_all=return_all)[0]
                        query_predictions = [prediction1 + prediction2 for prediction1, prediction2 in zip(query_predictions, predictions)]
                else:
                    predictions = evaluate.no_batch_predict(model_objs[0], datas, device=device, return_all=return_all, no_truth=True)[0]
                    for model_obj in model_objs[1:]:
                        predictions_ = evaluate.no_batch_predict(model_obj.to(device), datas, device=device, return_all=return_all, no_truth=True)[0]
                        predictions = [prediction1 + prediction2 for prediction1, prediction2 in zip(predictions, predictions_)]
                query_predictions = np.array([prediction/len(model_objs) for prediction in query_predictions])
            else:
                if model_type == 'scannet':
                    model_obj.reset_device(device)
                    query_predictions = [evaluate.test_predicts(inputs=inputs, model=model_obj, wrapper=model_wrappers[i], return_all=return_all)[0] for i, model_obj in enumerate(model_objs)]
                else:
                    query_predictions = [evaluate.no_batch_predict(model_obj.to(device), datas, device=device, return_all=return_all, no_truth=True)[0] for model_obj in model_objs]
        else:
            if model_type == 'scannet':
                model_obj.reset_device(device)
                query_predictions = evaluate.test_predicts(inputs=inputs, model=model_obj, wrapper=model_wrapper, return_all=return_all)[0]
            else:
                query_predictions = evaluate.no_batch_predict(model_obj.to(device), datas, device=device, return_all=return_all, no_truth=True)[0]

        if padded:
            query_predictions = wrappers.truncate_list_of_arrays(query_predictions, assembly_lengths)

    else:
        query_predictions = []
        for i in range(nqueries):
            # list of 8 inputs
            # s = time.time()
            inputs = wrappers.stack_list_of_arrays([pipeline.process_example(chain_obj=chain_obj, sequence=sequence)[0]
                                                    for chain_obj, sequence in zip(query_chain_objs[i], query_sequences[i])], padded=padded)
            # print('time to complete pipeline:', time.time()-s)
            if model_type != 'scannet':
                datas = dataloader.Baseline_Dataset(x=inputs, y=None, sample_weight=None, mode='val', num_layer_blocks=config_.num_layer_blocks, model_type=model_type)
            
            if multi_models:
                if aggregate_models:
                    if model_type != 'scannet':
                        predictions = evaluate.no_batch_predict(model_objs[0].to(device), datas, return_all=return_all, no_truth=True)[0]
                        for model_obj in model_objs[1:]:
                            predictions_ = evaluate.no_batch_predict(model_obj.to(device), datas, device=device, return_all=return_all, no_truth=True)[0]
                            predictions = [prediction1 + prediction2 for prediction1, prediction2 in zip(predictions, predictions_)]
                    else:
                        model_objs[0].reset_device(device)
                        predictions = evaluate.test_predicts(inputs=inputs, model=model_objs[0], wrapper=model_wrappers[0], return_all=return_all, inputs_id=i)[0]
                        for j, model_obj in enumerate(model_objs[1:]):
                            model_obj.reset_device(device)
                            predictions_ = evaluate.test_predicts(inputs=inputs, model=model_obj, wrapper=model_wrappers[j], device=device, return_all=return_all, inputs_id=i)[0]
                            predictions = [prediction1 + prediction2 for prediction1, prediction2 in zip(predictions, predictions_)]
                            
                    predictions = np.array([prediction/len(model_objs) for prediction in predictions])
                else:
                    if model_type == 'scannet':
                        model_obj.reset_device(device)
                        predictions = [evaluate.test_predicts(inputs=inputs, model=model_obj, wrapper=model_wrappers[j], return_all=return_all)[0] for j, model_obj in enumerate(model_objs)]
                    else:
                        predictions = [evaluate.no_batch_predict(model_obj.to(device), datas, device=device, return_all=return_all, no_truth=True)[0] for model_obj in model_objs]
            else:
                if model_type == 'scannet':
                    model_obj.reset_device(device)
                    predictions = evaluate.test_predicts(inputs=inputs, model=model_obj, wrapper=model_wrapper, return_all=return_all, inputs_id=i)[0]
                else:
                    query_predictions = evaluate.no_batch_predict(model_obj.to(device), datas, device=device, return_all=return_all, no_truth=True)[0]

            if padded:
                predictions = wrappers.truncate_list_of_arrays(predictions, sequence_lengths[i])

            if not aggregate_models:
                query_predictions.append(
                    [np.concatenate(prediction, axis=0) for prediction in predictions]
                )
            else:
                query_predictions.append(
                    np.concatenate(predictions, axis=0)
                )
        if not aggregate_models:
            query_predictions = [ [query_predictions[k][l] for k in range(len(query_predictions))] for l in range(len(query_predictions[0])) ]
    output_folder = predictions_folder + '/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    if output_predictions:
        for i in range(nqueries):
            res_ids = query_residue_ids[i]
            sequence = query_sequences[i]
            if not aggregate_models:
                predictions = [query_predictions_[i] for query_predictions_ in query_predictions]
            else:
                predictions = query_predictions[i]
            query_name = query_names[i]
            query_chain = query_chain_ids[i]
            query_chain_id_is_all = query_chain_id_is_alls[i]
            query_pdb = query_pdbs[i]
            file_is_cif = (pdb_file_locations[i][-4:] == '.cif')

            query_output_folder = output_folder+query_name
            if (len(query_pdb) == 4) & biounit:
                query_output_folder += '_biounit'

            if not query_chain_id_is_all:
                query_output_folder += '_(' + PDBio.format_chain_id(query_chain) + ')'

            if not assembly:
                query_output_folder += '_single'

            query_output_folder += '_%s' % model_name

            query_output_folder += '/'
            if not os.path.isdir(query_output_folder):
                os.mkdir(query_output_folder)

            if not aggregate_models:
                for prediction in predictions:
                    prediction = prediction[:,1]
                    csv_file = query_output_folder + 'predictions_' + query_name + '.csv'
                    chimera_file = query_output_folder + 'chimera_' + query_names[i]
                    annotated_pdb_file = query_output_folder + 'annotated_' + query_names[i] + ('.cif' if file_is_cif else '.pdb')
                    
                    write_predictions(csv_file, res_ids,sequence, prediction)
                    if predict_from_pdb & (prediction.ndim == 1):
                        if output_chimera == 'script':
                            chimera.show_binding_sites(query_pdbs[i], csv_file, chimera_file, biounit=biounit, directory='',thresholds=chimera_thresholds)
                        elif output_chimera == 'annotation':
                            mini = 0.5
                            maxi = 2.5
                            chimera.annotate_pdb_file(pdb_file_locations[i], csv_file, annotated_pdb_file, output_script=True, mini=mini, maxi=maxi,version='surface' if assembly else 'default')
            else:
                csv_file = query_output_folder + 'predictions_' + query_name + '.csv'
                chimera_file = query_output_folder + 'chimera_' + query_names[i]
                annotated_pdb_file = query_output_folder + 'annotated_' + query_names[i] + ('.cif' if file_is_cif else '.pdb')
                
                write_predictions(csv_file, res_ids, sequence, predictions)
                if predict_from_pdb & (predictions.ndim == 1):
                    if output_chimera == 'script':
                        chimera.show_binding_sites(
                            query_pdbs[i], csv_file, chimera_file, biounit=biounit, directory='',thresholds=chimera_thresholds)
                    elif output_chimera == 'annotation':
                        mini = 0
                        maxi = chimera_thresholds[-1]

                        chimera.annotate_pdb_file(pdb_file_locations[i], csv_file, annotated_pdb_file, output_script=True, mini=mini, maxi=maxi,version='surface' if assembly else 'default')
            print('File at:', annotated_pdb_file)
    if output_format == 'dictionary':
        if not aggregate_models:
            query_dictionary_predictions = [PDB_processing.make_values_dictionary(query_residue_ids[k], [query_predictions[l][k] for l in range(len(query_predictions))])
                                            for k in range(len(query_residue_ids))]
        else:
            query_dictionary_predictions = [PDB_processing.make_values_dictionary(query_residue_id,query_prediction) for query_residue_id,query_prediction in zip(query_residue_ids,query_predictions)]
        return query_pdbs, query_names, query_dictionary_predictions
    else:
        if not aggregate_models:
            query_predictions = [
                [query_predictions[i][j] for i in range(len(query_predictions))] for j in range(len(query_predictions[0]))]
        return query_pdbs, query_names, query_predictions, query_residue_ids, query_sequences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict binding sites in PDB files using Geometric Neural Network')
    setting = cfg.Model_SetUp()
    
    
    parser.add_argument('input',  type=str,
                   help='Three input formats. i) A pdb id (1a3x)\
                   ii) Path to pdb file (structures/1a3x.pdb)\
                   iii) Path to text file containing list of pdb files (one per line) (1a3x \n 2kho \n ...) \
                   For performing prediction only on specfic chains, append "_" and the list of chains. (e.g. 1a3x_AB)')
    
    parser.add_argument('--name', dest='name',
                        default='',
                        help='Input name')
    
    parser.add_argument('--version', dest='version', default=-1, type=int, help='Version number of the model')
                        
    parser.add_argument('--predictions_folder', dest='predictions_folder',
                        default='',
                        help='Input name')
    parser.add_argument('--structure_folder', dest='structure_folder',
                        default='',
                        help='Path to structure folder')

    parser.add_argument('--interaction', dest='interaction', default='PPI', choices=['PPI', 'PPeI', 'PAI', 'PAI1', 'PAI2', 'PAI3', 'PAI4', 'PAI5'], help='Prediction mode (PPI, PPeI, PAI)')
    parser.add_argument('--model_type', dest='model_type', default='scannet', choices=['scannet', 'unet', 'rcnn'],
                        help='Choose the network to predict the binding sites (for the latter two mode types, only S2Site on PPI has been provided)')
    
    parser.add_argument('--tl', dest='tl', action='store_true', default=False, 
                        help='Use the PPeI/PAI network that is trained from transfer learning or random initialization')
    
    parser.add_argument('--assembly', dest='assembly', action='store_const',
                 const=True, default=False,
                 help = 'Perform prediction from single chains or from biological assemblies')

    parser.add_argument('--permissive', dest='permissive', action='store_const',
                        const=True, default=True, help='Permissive prediction')

    parser.add_argument('--return_all_class', dest='return_all', action='store_true', default=False, 
                        help='Choose output layer')

    parser.add_argument('--no_biounit',  dest='biounit',action='store_const',
                   const=False, default=True,
                   help='Predict from pdb/cif file (default=predict from biounit file)')
    parser.add_argument('--downloadPDB', dest='download_noFound', action='store_true', default=False,
                        help='Retrieve the PDB online if it is not found in the folder')
    
    parser.add_argument('--ncores', dest='cores', type=int, default=8, help='Number of CPUs to program')
    parser.add_argument('--gpu_id', dest='device', type=str, default='0', help='GPU id to program')
    
    args = parser.parse_args()
    
    os.environ["MKL_NUM_THREADS"] = "%s" % args.cores
    os.environ["NUMEXPR_NUM_THREADS"] = "%s" % args.cores
    os.environ["OMP_NUM_THREADS"] = "%s" % args.cores
    os.environ["OPENBLAS_NUM_THREADS"] = "%s" % args.cores
    os.environ["VECLIB_MAXIMUM_THREADS"] = "%s" % args.cores
    os.environ['NUMBA_DEFAULT_NUM_THREADS'] = "%s" % args.cores
    os.environ["NUMBA_NUM_THREADS"] = "%s" % args.cores
    
    setting.reset_model()
    setting.reset_environment(device=args.device)
    config = setting.get_setting()
    
    config.use_esm = True
    config.download_noFound = args.download_noFound
    
    pipeline = pipelines.S2SitePipeline(
        with_aa=True,
        with_atom=True,
        aa_features='esm',
        atom_features='valency',
        aa_frames='triplet_sidechain',
        Beff=500,
        device=config.device
    )

    input = args.input
    query_pdbs = []
    query_chain_ids = []
    if '.txt' in input:
        with open(input, 'r') as f:
            for line in f:
                pdb, chain_ids = PDBio.parse_str(line[:-1].strip())
                query_pdbs.append(pdb)
                query_chain_ids.append(chain_ids)
    else:
        query_pdbs, query_chain_ids = PDBio.parse_str(input)

    if args.name != '':
        query_names = [args.name]
    else:
        query_names = None
    
    if len(args.predictions_folder) > 0:
        config.predicts = args.predictions_folder
        
    if len(args.structure_folder) > 0:
        config.structures_folder = args.structure_folder
    version = 0 if args.version == -1 else args.version
    
    if args.interaction == 'PPI':
        chimera_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        if args.model_type == 'scannet':
            model_name = f'S2Site_PPI_3Bv{version}'
            model_id = f'S2Site_PPI_3Bv{version}_PPBS.pth'
        elif args.model_type == 'rcnn':
            model_name = f'RCNN_PPI_v{version}'
            model_id = f'Baseline_RCNNv{version}wo2_2_0_PPBS.pth'
        elif args.model_type == 'unet':
            model_name = f'UNet_PPI_v{version}'
            model_id = f'Baseline_UNetv{version}_2_PPBS.pth'
            
    elif args.interaction == 'PPeI':
        chimera_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        version = 0 if args.version == -1 else args.version
        if args.tl:
            prefix = f'tlv{version}'
            suffix = 'PPBSTLPPeBS'
        else:
            prefix = f'PPeI_3Bv{version}'
            suffix = 'PPeBS'
        model_name = f'S2Site_{prefix}_{suffix}'
        model_id = f'S2Site_{prefix}_{suffix}.pth'
        
    elif args.interaction == 'PAI':
        chimera_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        if args.tl:
            prefix =  f'tlv{version}'
            suffix = 'PPBSTLBCE'
        else:
            prefix = f'PAI_3Bv{version}'
            suffix = 'BCE'
        model_name = [f'S2Site_{prefix}_{index+1}' for index in range(5)]
        model_id = [f'S2Site_{prefix}_{suffix}_{index+1}.pth' for index in range(5)]

    elif args.interaction[:-1] == 'PAI': # epitope1, epitope2, epitope3, epitope4, epitope5
        fold = args.interaction[-1]
        chimera_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        if args.tl:
            prefix =  f'tlv{version}'
            suffix = 'PPBSTLBCE'
        else:
            prefix = f'PAI_3Bv{version}'
            suffix = 'BCE'
            
        model_name = f'S2Site_{prefix}_{fold}'
        model_id = f'S2Site_{prefix}_{suffix}_{fold}.pth'
        
    predict_interface_residues(
        config=config,
        query_pdbs=query_pdbs,
        query_chain_ids=query_chain_ids,
        query_names=query_names,
        pipeline=pipeline,
        model=model_id,
        model_name=model_name,
        model_type=args.model_type,
        biounit=args.biounit,
        assembly=args.assembly,
        permissive=args.permissive,
        chimera_thresholds=chimera_thresholds,
        return_all=args.return_all,
        device=config.device
    )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
