from set_seed import *
from data import *
from model import *
from utils import *
from torch.optim import AdamW

_exp_name = None
_base_dir = "."

def parse_args():
	"""arguments"""
	config = {
		"seed": 100,
		"data_dir": _base_dir + "/Dataset",
		"save_path": _base_dir + f"/{_exp_name}",
		"batch_size": 64,
		"lr": 1e-3,
		'm': 0.3,
		"n_workers": 8,
		"valid_steps": 2000,
		"warmup_steps": 6000,
		"save_steps": 10000,
		"total_steps": 300000,
		"last_step": -1
	}

	return config


def main(
	seed,
	data_dir,
	save_path,
	batch_size,
	lr,
	m,
	n_workers,
	valid_steps,
	warmup_steps,
	total_steps,
	save_steps,
	last_step
):
	"""Main function."""
	if not os.path.exists(os.path.join(_base_dir,_exp_name)):
		os.mkdir(os.path.join(_base_dir,_exp_name))

	set_seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
	train_iterator = iter(train_loader)
	print(f"[Info]: Finish loading data!",flush = True)

	model = Classifier(n_spks=speaker_num).to(device)
	criterion = AMSoftmaxCrossEntropy(m)
	optimizer = AdamW(model.parameters(), lr=lr)
	scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
	if last_step > -1:
		model.load_state_dict(torch.load(os.path.join(save_path,'model.ckpt')))
		optimizer.load_state_dict(torch.load(os.path.join(save_path,'optim.ckpt')))
		scheduler.load_state_dict(torch.load(os.path.join(save_path,'sched.ckpt')))
		print(f'Valid acc: {valid(valid_loader, model, criterion, device):.4f}.')
	print(f"[Info]: Finish creating model!",flush = True)

	best_accuracy = -1.0
	best_state_dict_model = None
	best_state_dict_optim = None
	best_state_dict_sched = None
	train_acc = []

	pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

	for step in range(last_step+1,total_steps):
		# Get data
		try:
			batch = next(train_iterator)
		except StopIteration:
			train_iterator = iter(train_loader)
			batch = next(train_iterator)

		loss, accuracy = model_fn(batch, model, criterion, device)
		batch_loss = loss.item()
		batch_accuracy = accuracy.item()
		train_acc.append(batch_accuracy)

		# Updata model
		loss.backward()
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

		# Log
		pbar.update()
		pbar.set_postfix(
			loss=f"{batch_loss:.2f}",
			accuracy=f"{batch_accuracy:.2f}",
			lr=scheduler.get_last_lr()[0],
			step=step + 1,
		)

		# Do validation
		if (step + 1) % valid_steps == 0:
			pbar.close()

			valid_accuracy = valid(valid_loader, model, criterion, device)

			# keep the best model
			if valid_accuracy > best_accuracy:
				best_accuracy = valid_accuracy
				best_state_dict_model = model.state_dict()
				best_state_dict_optim = optimizer.state_dict()
				best_state_dict_sched = scheduler.state_dict()

			pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

			pbar.write(f"Train acc: {np.mean(train_acc):.4f}; test acc: {valid_accuracy:.4f}.")
			train_acc.clear()

		# Save the best model so far.
		if (step + 1) % save_steps == 0 and best_state_dict_model is not None:
			torch.save(best_state_dict_model, os.path.join(save_path, 'model.ckpt'))
			torch.save(best_state_dict_optim, os.path.join(save_path, 'optim.ckpt'))
			torch.save(best_state_dict_sched, os.path.join(save_path, 'sched.ckpt'))
			pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

	pbar.close()


if __name__ == "__main__":
	main(**parse_args())