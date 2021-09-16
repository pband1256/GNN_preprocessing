import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score

import multi_utils as utils
import model
from data_handler import construct_loader

#####################
#     CONSTANTS     #
#####################
TEST_NAME='Test'

#######################
#     EXPERIMENT      #
#######################

def train_one_epoch(net,
                    criterion,
                    optimizer,
                    args,
                    experiment_dir,
                    train_loader,
                    ):
  net.train()
  nb_train = len(train_loader) * args.batch_size
  epoch_loss = 0

  pred_y = np.zeros((nb_train, output_dim))
  true_y = np.zeros((nb_train, output_dim))
  for i, batch in enumerate(train_loader):
    X, y, w, adj_mask, batch_nb_nodes, _, _, _, _ = batch

    # Pick corresponding labels
    y = y[:,INDEX:INDEX+output_dim]

    X, y, w, adj_mask, batch_nb_nodes = X.to(device), y.to(device), w.to(device), adj_mask.to(device), batch_nb_nodes.to(device)
    optimizer.zero_grad()
    out = net(X, adj_mask, batch_nb_nodes)
    loss = criterion(out, y).cuda()
    loss.backward()
    optimizer.step()
    
    beg =     i * args.batch_size
    end = (i+1) * args.batch_size
    pred_y[beg:end,:] = out.data.cpu().numpy()
    true_y[beg:end,:] = y.data.cpu().numpy()
    
    epoch_loss += loss.item()
    # Print running loss about 10 times during each epoch
    if (((i+1) % (len(train_loader)//10)) == 0):
      nb_proc = (i+1)*args.batch_size
      logging.info("  {:5d}: {:.9f}".format(nb_proc, epoch_loss/nb_proc))

  epoch_loss /= nb_train
  logging.info("Train: loss {:>.3E}".format(epoch_loss))
  return (epoch_loss)


def train(
          net,
          criterion,
          args, 
          experiment_dir, 
          multi_train_loader, 
          valid_loader
          ):
  optimizer = torch.optim.Adamax(net.parameters(), lr=args.lrate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.patience)
  # Nb epochs completed tracked in case training interrupted
  for i in range(args.nb_epochs_complete, args.nb_epoch):
    # Update learning rate in optimizer
    t0 = time.time()
    logging.info("\nEpoch {}".format(i+1))
    logging.info("Learning rate: {0:.3g}".format(args.lrate))
    
    # Switching between training sets
    train_loader = multi_train_loader[i % len(multi_train_loader)]
    logging.info("Training on "+args.train_file[i % len(multi_train_loader)])

    train_stats = train_one_epoch(net,
                                  criterion,
                                  optimizer,
                                  args,
                                  experiment_dir,
                                  train_loader)
    if (((i+1) % len(multi_train_loader)) == 0):
      val_stats = evaluate(net, criterion, experiment_dir, args,
                            valid_loader, 'Valid')                          
      utils.track_epoch_stats(i/len(args.train_file), args.lrate, 0, train_stats, val_stats, experiment_dir)

      # Update learning rate, remaining nb epochs to train
      scheduler.step(val_stats)

      # Track best model performance
      if (val_stats < args.best_loss):
        logging.warning("Best performance on valid set.")
        nb_files = len(args.train_file)
        args.best_loss = float(val_stats)
        utils.plot_loss(train_stats, val_stats, i//nb_files, experiment_dir)
        utils.update_best_plots(experiment_dir)
        utils.save_best_model(experiment_dir, net)
        utils.save_best_scores(i/nb_files, val_stats, experiment_dir)
        utils.save_epoch_model(experiment_dir, net)

    args.lrate = optimizer.param_groups[0]['lr']
    args.nb_epochs_complete += 1

    utils.save_args(experiment_dir, args)
    logging.info("Epoch took {} seconds.".format(int(time.time()-t0)))
    
    if args.lrate < 10**-4:
        logging.warning("Minimum learning rate reached.")
        break

  logging.warning("Training completed.")


def evaluate(net,
             criterion,
             experiment_dir,
             args,
             valid_loader,
             plot_name):
             
    net.eval()
    epoch_loss = 0
    nb_batches = len(valid_loader)
    nb_eval = nb_batches * args.batch_size
    # Track samples by batches for scoring
    pred_y = np.zeros((nb_eval, output_dim))
    true_y = np.zeros((nb_eval, output_dim))
    evt_id = []
    f_name = []
    E_name = []
    r_name = []
    logging.info("Evaluating {} {} samples.".format(nb_eval,plot_name))
    with torch.autograd.no_grad():
        for i, batch in enumerate(valid_loader):
            X, y, w, adj_mask, batch_nb_nodes, evt_ids, evt_names, energy, reco = batch
            # Pick corresponding labels
            y = y[:,INDEX:INDEX+output_dim]
            # Remove one-hot encoding
            #y_org = y.to(device)
            #y = np.argmax(y,axis=1).long()
            X, y, w, adj_mask, batch_nb_nodes = X.to(device), y.to(device), w.to(device), adj_mask.to(device), batch_nb_nodes.to(device)
            out = net(X, adj_mask, batch_nb_nodes)

            loss = criterion(out, y).cuda()
            epoch_loss += loss.item() 
            # Track predictions, truth, weights over batches
            beg =     i * args.batch_size
            end = (i+1) * args.batch_size
            pred_y[beg:end,:] = out.data.cpu().numpy()
            true_y[beg:end,:] = y.data.cpu().numpy()
            if plot_name==TEST_NAME:
                evt_id.extend(evt_ids)
                f_name.extend(evt_names)
                E_name.extend(energy)
                r_name.extend(reco)

            # Print running loss 2 times 
            if (((i+1) % (nb_batches//2)) == 0):
                nb_proc = (i+1)*args.batch_size
                logging.info("  {:5d}: {:.9f}".format(nb_proc, epoch_loss/nb_proc))

    # Score predictions, save plots, and log performance
    epoch_loss /= nb_eval # Normalize loss
    logging.info("{}: loss {:>.3E}".format(plot_name, epoch_loss))

    if plot_name == TEST_NAME:
        E_min = np.min(E_name)
        E_max = np.max(E_name) 
        # Resolution plot
        '''
        if args.regr_mode == 'direction':
            utils.plot_energy_slices(true_y[:,0], pred_y[:,0], regr_mode='zenith', use_fraction=True, bins=20, minenergy=E_min, maxenergy=E_max, savefolder=experiment_dir)
            utils.plot_energy_slices(true_y[:,1], pred_y[:,1], regr_mode='azimuth', use_fraction=True, bins=20, minenergy=E_min, maxenergy=E_max, savefolder=experiment_dir)
        elif args.regr_mode == 'direction_cart':
            utils.plot_energy_slices(true_y[:,0], pred_y[:,0], regr_mode='x', use_fraction=True, bins=20, minenergy=E_min, maxenergy=E_max, savefolder=experiment_dir)
            utils.plot_energy_slices(true_y[:,1], pred_y[:,1], regr_mode='y', use_fraction=True, bins=20, minenergy=E_min, maxenergy=E_max, savefolder=experiment_dir)
            utils.plot_energy_slices(true_y[:,2], pred_y[:,2], regr_mode='z', use_fraction=True, bins=20, minenergy=E_min, maxenergy=E_max, savefolder=experiment_dir)
        elif args.regr_mode == 'energy':
            utils.plot_energy_slices(true_y, pred_y, regr_mode='energy', use_fraction=True, old_reco=r_name, bins=20, minenergy=E_min, maxenergy=E_max, savefolder=experiment_dir)
        '''
        utils.plot_reg_hist(true_y, pred_y, experiment_dir, regr_mode=args.regr_mode)
        utils.save_test_scores(nb_eval, epoch_loss, experiment_dir)
        utils.save_preds(evt_id, f_name, E_name, pred_y, true_y, experiment_dir)
    return (epoch_loss)


def main():
  input_dim = 9
  spatial_dim = [0,1,2]
  args = utils.read_args()

  global output_dim
  global INDEX
  # Labels: [energy, zenith, azimuth, x, y, z]
  # Output: energy
  if args.regr_mode == 'energy':
      output_dim = 1
      INDEX = 0
  # Output: zenith, azimuth
  elif args.regr_mode == 'direction':
      output_dim = 2
      INDEX = 1
  # Output: x, y, z
  elif args.regr_mode == 'direction_cart':
      output_dim = 3
      INDEX = 3
  else:
      assert True, 'Regression quantity not defined'

  experiment_dir = utils.get_experiment_dir(args.name, args.run)
  utils.initialize_experiment_if_needed(experiment_dir, args.evaluate)
  # Logger will print to stdout and logfile
  utils.initialize_logger(experiment_dir)

  # Optionally restore arguments from previous training
  # Useful if training is interrupted
  if not args.evaluate:
    try:
      args = utils.load_args(experiment_dir)
    except:
      args.best_loss = np.Inf
      args.nb_epochs_complete = 0 # Track in case training interrupted
      utils.save_args(experiment_dir, args) # Save initial args

  net = utils.create_or_restore_model(
                                    experiment_dir, 
                                    args.nb_hidden, 
                                    args.nb_layer,
                                    input_dim,
                                    output_dim,
                                    spatial_dim
                                    )
  if not torch.cuda.is_available():
    raise Exception('No GPU available.')
    
  net = net.cuda()
  logging.warning("Training on GPU")
  logging.info("GPU type:\n{}".format(torch.cuda.get_device_name(0)))
  logging.info("Number of GPU: {}".format(torch.cuda.device_count()))
  # Setting default tensor type to cuda
  global device
  device = torch.device('cuda')

  # Multiclass loss function
  criterion = nn.MSELoss()  #CrossEntropyLoss()
  if not args.evaluate:
    assert (args.train_file != None)
    assert (args.val_file   != None)
    multi_train_loader = []
    for file in args.train_file:
      train_loader = construct_loader(
                                file,
                                args.nb_train,
                                args.batch_size,
                                shuffle=True)
      multi_train_loader.append(train_loader)
    valid_loader = construct_loader(args.val_file,
                                    args.nb_val,
                                    args.batch_size)
    logging.info("Training on {} samples.".format(
                                          len(multi_train_loader)*len(train_loader)*args.batch_size))
    logging.info("Validate on {} samples.".format(
                                          len(valid_loader)*args.batch_size))
    train(
              net,
              criterion,
              args,
              experiment_dir,
              multi_train_loader,
              valid_loader
         )

  # Perform evaluation over test set
  try:
    net = utils.load_best_model(experiment_dir)
    logging.warning("\nBest model loaded for evaluation on test set.")
  except:
    logging.warning("\nCould not load best model for test set. Using current.")
  assert (args.test_file != None)
  test_loader = construct_loader(args.test_file,
                                 args.nb_test,
                                 args.batch_size)
  test_stats = evaluate(net,
                        criterion,
                        experiment_dir,
                        args,
                        test_loader,
                        TEST_NAME)

if __name__ == "__main__":
  main()

