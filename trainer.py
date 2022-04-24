from utils import AverageMeter,ProgressMeter
import torch
import time
from torch.autograd import Variable

def train_dir(train_loader, model, nce_criterion, mse_criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    nce_losses = AverageMeter('NCE Loss', ':.4e')
    mse_losses = AverageMeter('MSE Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, nce_losses,mse_losses,losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.mode.lower() == "di":
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
            output, target = model(im_q=images[0], im_k=images[1])
            loss = nce_criterion(output, target)
            nce_losses.update(loss.item(), images[0].size(0))
            losses.update(loss.item(), images[0].size(0))
        else:
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
                images[2] = images[2].cuda(args.gpu, non_blocking=True)
            output, target, rec_output = model(im_q=images[0], im_k=images[1])
            nce_loss = nce_criterion(output, target)
            mse_loss = mse_criterion(rec_output, images[2])
            loss = args.contrastive_weight * nce_loss + args.mse_weight * mse_loss

            nce_losses.update(nce_loss.item(), images[0].size(0))
            mse_losses.update(mse_loss.item(), images[0].size(0))
            losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate_dir(val_loader, model, nce_criterion, mse_criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    nce_losses = AverageMeter('NCE Loss', ':.4e')
    mse_losses = AverageMeter('MSE Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, nce_losses,mse_losses,losses],
        prefix="Validation: ")

    model.eval()
    counter = torch.zeros((2,), device=torch.device(f'cuda:{args.rank}'))

    end = time.time()
    for i, (images) in enumerate(val_loader):
        with torch.no_grad():
        # measure data loading time
            data_time.update(time.time() - end)
            if args.mode.lower() == "di":
                if args.gpu is not None:
                    images[0] = images[0].cuda(args.gpu, non_blocking=True)
                    images[1] = images[1].cuda(args.gpu, non_blocking=True)
                output, target = model(im_q=images[0], im_k=images[1])
                loss = nce_criterion(output, target)
                nce_losses.update(loss.item(), images[0].size(0))
                losses.update(loss.item(), images[0].size(0))
            else:
                if args.gpu is not None:
                    images[0] = images[0].cuda(args.gpu, non_blocking=True)
                    images[1] = images[1].cuda(args.gpu, non_blocking=True)
                    images[2] = images[2].cuda(args.gpu, non_blocking=True)
                output, target, rec_output = model(im_q=images[0], im_k=images[1])
                nce_loss = nce_criterion(output, target)
                mse_loss = mse_criterion(rec_output, images[2])
                loss = args.contrastive_weight * nce_loss + args.mse_weight * mse_loss

                nce_losses.update(nce_loss.item(), images[0].size(0))
                mse_losses.update(mse_loss.item(), images[0].size(0))
                losses.update(loss.item(), images[0].size(0))

            counter[0] += loss.item()
            counter[1] += 1

        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return counter

def train_dira(train_loader, generator, nce_criterion, mse_criterion, adversarial_criterion, optimizer_G, epoch, args, discriminator,optimizer_D,D_output_shape):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    nce_losses = AverageMeter('NCE Loss', ':.4e')
    mse_losses = AverageMeter('MSE Loss', ':.4e')
    g_losses = AverageMeter('Adversarial G Loss', ':.4e')
    d_losses = AverageMeter('Discriminator Loss', ':.4e')
    losses = AverageMeter('Generator Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, nce_losses,mse_losses,g_losses,d_losses,losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    generator.train()

    end = time.time()
    Tensor = torch.cuda.FloatTensor
    for i, (images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        valid = Variable(Tensor(images[0].shape[0], *D_output_shape).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(images[0].shape[0], *D_output_shape).fill_(0.0), requires_grad=False)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)

        # -----------------
        #  Train Generator
        # -----------------

        # compute output
        output, target, rec_output = generator(im_q=images[0], im_k=images[1])
        nce_loss = nce_criterion(output, target)
        mse_loss = mse_criterion(rec_output, images[2])
        g_adv = adversarial_criterion(discriminator(rec_output), valid)

        g_loss = args.contrastive_weight * nce_loss + args.mse_weight * mse_loss + args.adv_weight * g_adv

        nce_losses.update(nce_loss.item(), images[0].size(0))
        mse_losses.update(mse_loss.item(), images[0].size(0))
        g_losses.update(g_adv.item(), images[0].size(0))
        losses.update(g_loss.item(), images[0].size(0))

        # compute gradient and do SGD step

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_criterion(discriminator(images[2]), valid)
        fake_loss = adversarial_criterion(discriminator(rec_output.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        d_losses.update(d_loss.item(), images[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate_dira(val_loader, model, nce_criterion, mse_criterion,adversarial_criterion, epoch, args,discriminator,D_output_shape):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    nce_losses = AverageMeter('NCE Loss', ':.4e')
    mse_losses = AverageMeter('MSE Loss', ':.4e')
    g_losses = AverageMeter('Adversarial G Loss', ':.4e')
    losses = AverageMeter('Generator Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, nce_losses,mse_losses,losses],
        prefix="Validation: ")

    # switch to train mode
    model.eval()
    counter = torch.zeros((2,), device=torch.device(f'cuda:{args.rank}'))

    end = time.time()
    Tensor = torch.cuda.FloatTensor
    for i, (images) in enumerate(val_loader):
        with torch.no_grad():
        # measure data loading time
            data_time.update(time.time() - end)

            valid = Variable(Tensor(images[0].shape[0], *D_output_shape).fill_(1.0), requires_grad=False)

            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
                images[2] = images[2].cuda(args.gpu, non_blocking=True)


            # compute output
            output, target, rec_output = model(im_q=images[0], im_k=images[1])
            nce_loss = nce_criterion(output, target)
            mse_loss = mse_criterion(rec_output, images[2])
            g_adv = adversarial_criterion(discriminator(rec_output), valid)
            loss = args.contrastive_weight * nce_loss + args.mse_weight * mse_loss + args.adv_weight * g_adv
            nce_losses.update(nce_loss.item(), images[0].size(0))
            mse_losses.update(mse_loss.item(), images[0].size(0))
            g_losses.update(g_adv.item(), images[0].size(0))
            losses.update(loss.item(), images[0].size(0))

            counter[0] += loss.item()
            counter[1] += 1

        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return counter
