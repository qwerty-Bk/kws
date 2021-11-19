from tqdm import tqdm
import torch.nn.functional as F
import torch


def distill_train_epoch(teacher, student, opt, loader, log_melspec,
                        device, temp, alpha):
    teacher.eval()
    student.train()
    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(batch) / temp
        student_logits = student(batch)

        teacher_labels = F.softmax(teacher_logits, dim=-1)
        probs = F.softmax(student_logits, dim=-1)

        loss_on_labels = F.cross_entropy(student_logits, labels)
        loss_on_teacher = F.kl_div(F.log_softmax(student_logits / temp, dim=-1), teacher_labels / temp, reduction="batchmean")

        loss = loss_on_teacher * alpha * temp ** 2 + (1 - alpha) * loss_on_labels

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5)

        opt.step()

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return acc
