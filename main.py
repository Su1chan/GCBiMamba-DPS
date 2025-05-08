import math
import numpy as np
import torch
import os
from causal import SessionDataset, collate_fn, Recommender, Mamba4Rec
import pickle
from pathlib import Path
import time
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# from pyinstrument import Profiler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():

    # # todo
    # profiler = Profiler()
    # profiler.start()

    #! 处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default='p = 0.3', help='comments')
    parser.add_argument("--data", default='lastfm')
    parser.add_argument("--split", default=0, type=int)

    parser.add_argument("--load", default=False)
    parser.add_argument("--save_path", default='model_1.pkl')

    parser.add_argument("--sampling", default='recent')
    parser.add_argument("--sample_size", default=10, type=int)
    parser.add_argument("--max_len_recent", default=10, type=int)
    parser.add_argument("--c", default=11, type=float)
    parser.add_argument("--b1", default=0.2, type=float)
    parser.add_argument("--b2", default=1, type=float)


    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--gamma", default=1, type=float)

    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--n_epoch", default=77, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--l2", default=1e-5, type=float)
    parser.add_argument("--eval_every", default=5,
                        help='evalutate every 5 epochs', type=int)
    parser.add_argument("--no_cuda", default=False)
    parser.add_argument("--device", default='cuda:0')
    # parser.add_argument("--seed", default=2022)
    parser.add_argument("--seed", default=2022)


    parser.add_argument("--weighting", default='div')
    parser.add_argument("--user_key", default='user')
    parser.add_argument("--item_key", default='item')
    parser.add_argument("--time_key", default='timestamp')
    parser.add_argument("--session_key", default='sessionId')
    args = parser.parse_args()

    #! 初始化其他参数
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    PRJ_PATH = Path(__file__).resolve().parent
    DATA_PATH = PRJ_PATH / 'data'

    #! dataset处理
    # stime = time.time()
    with open(DATA_PATH/f'fm_{args.data}_{args.split}.pkl', 'rb') as f:
    # with open("D:\Edge\COCO-SBRS-main\COCO-SBRS-main\data\fm_delicious_1.pkl", 'rb') as f:
        trainset, testset = pickle.load(f)

    # todo test code
    load_dataset = True
    if not load_dataset:
        dataset = SessionDataset(trainset, testset, user_key=args.user_key, item_key=args.item_key, session_key=args.session_key,
                                 time_key=args.time_key, max_len_recent=args.max_len_recent, device=device,
                                 sample_size=args.sample_size, sampling=args.sampling)
        with open(f'dataset_{args.data}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            print('write dataset to disk')
    else:
        print('load data')
        with open(f'dataset_{args.data}.pkl', 'rb') as f:
            dataset = pickle.load(f)
    # profiler.stop()
    # profiler.print()
    # exit()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda p: collate_fn(p, device=device))

    # model = Recommender(args.dim, dataset.user_number,
    #                     dataset.item_number, device=device)
    # model.user_item_freq = dataset.user_item_freq
    model = Mamba4Rec()
    model = model.to(device)


    # todo load model from file
    # if args.load and os.path.exists(args.save_path):
    #     model.load_state_dict(torch.load(
    #         args.save_path, map_location=device))
    #     return
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=args.lr, weight_decay=args.l2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)

    n_batch = (dataset.session_number // args.batch_size) + 1

    best_recall20 = 0
    best_test = ''
    # log_file = open(PRJ_PATH/'results' /
    #                 f'my-{args.data}-ss{args.sample_size}-c{args.c:.1f}-b{args.b1:.1f}-{args.d}.txt', 'w')

    log_file = open(PRJ_PATH / 'results' /
                    f'my-{args.data}-c{args.c:.1f}-b{args.b2:.1f}--{args.d}.txt', 'w')

    for k, v in args.__dict__.items():
        print(f'{str(k)}:  {str(v)}', file=log_file, flush=True)
        print(f'{str(k)}:  {str(v)}', flush=True)
    patience_idx = 0

    # alpha, beta, gamma = 1, 1, 0.25

    #! train
    for it in range(args.n_epoch):
        print(f'Epoch: {(it + 1)} starts...')
        print(f'Epoch: {(it + 1)}', file=log_file, flush=True)
        loss_list = []
        loss_plist = []
        loss_clist = []
        loss_contrast_list = []
        idx_batch = 1
        for users, session, recent, all_item in train_loader:
            model.train()
            optimizer.zero_grad()

            # filtering samples have empty recent item set
            users, session, recent, all_item = mask_no_recent(
                users, session, recent, all_item)
            items_to_score, map_label = session.unique(return_inverse=True)
            # nonzero = items_to_score != 0
            # items_to_score = items_to_score[nonzero]
            # scores, hu, hc, attn = model(
            #     users, session[:, :-1], recent, items_to_score)

            scores, anchor, positive, negative, hu, hc, attn = model(
                  users, session[:, :-1], recent, items_to_score)
            # print(anchor.shape)
            # print(positive.shape)
            # print(negative.shape)


            loss_predict = model.loss_predict1(
                scores, session, map_label)
            loss = loss_predict
            loss_plist.append(loss_predict.item())

            if args.c > 0:
                # loss_contrst = model.loss_contrastive(
                #     all_item, session[:, :-1], hu, hc)
                # loss = loss + model.c * loss_contrst
                # loss_clist.append(loss_contrst.item())

                # loss_causal = model.loss_causal(    # 伪标签
                #     attn, session[:, :-1], all_item, items_to_score
                # )
                loss_causal = model.loss_causal1(  # 伪标签
                    attn, session[:, :-1], all_item, items_to_score, users
                )
                loss_contrast = model.contrast_loss(anchor, positive, negative)
                # loss = loss + args.c * loss_causal
                loss = args.alpha * loss_predict + args.beta * loss_causal + args.gamma * loss_contrast
                loss_clist.append(loss_causal.item())
                loss_contrast_list.append(loss_contrast.item())

            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            print(f'\rbatch {idx_batch}/{n_batch} loss: {loss.item():.5f}',
                  end='')  # loss1: {loss_predict.item():.5f} loss2: {loss_contrst.item():.5f}

            idx_batch += 1



            assert torch.isnan(model.item_embedding.weight).sum() == 0
            assert torch.isnan(model.user_embedding.weight).sum() == 0
        epoch_loss = np.array(loss_list).mean()
        print(
            f'\nEpoch: {(it + 1)} total loss: {epoch_loss}, loss1 {np.array(loss_plist).mean()}, loss2 {np.array(loss_clist).mean()}, loss3 {np.array(loss_contrast_list).mean()}')
        print('====================================================================')

        #! evaluation
        val_ndcg_list = []  # 存储每个fold的NDCG@20
        test_ndcg_list = []  # 存储每个fold的NDCG@20
        coco_ndcg_list = [...]  # 例如 [0.401, 0.398, 0.405, 0.392, 0.407]
        l = []  # 每个测试session，百分之多少的target item在item to predict中
        if True:  # it > 0 and (it+1)//args.eval_every == 0:
            print('Start evaluation...')
            print(f'total: {len(dataset.session_ids_test)} sessions to test.')
            tester_val = Tester(k_list=[5, 10, 20])
            tester_tst = Tester(k_list=[5, 10, 20])
            for idx, test_session in tqdm(enumerate(dataset.session_ids_test)):
                session_item, users, recent, similarity, items_to_boost = dataset.next_test_session(
                    test_session, device)

                session_context = session_item[:-1].unsqueeze(0)
                with torch.no_grad():
                    scores = model.test_session(users, session_context,
                                                recent, similarity)
                    scores = scores * args.b2
                    # scores = scores + float(args.b1) * \
                    #     items_to_boost[np.newaxis, :]
                    boost_score = float(args.b1) * \
                        items_to_boost[np.newaxis, :]
                    # if float(args.b2) > 0:  # boost context item neighbors
                    #     items_to_boost2 = dataset.next_test_session2(
                    #         test_session)  # context boost item
                    #     boost_score = boost_score + \
                    #         float(args.b2) * items_to_boost2
                    #     boost_score[boost_score > max(float(args.b1), float(args.b2))] = max(
                    #         float(args.b1), float(args.b2))
                    scores = scores + boost_score

                    sorted_score_index = scores.argsort(1)[:, ::-1]
                    predicted_seq = sorted_score_index + 1  # todo 只适用于预测所有item
                    # predicted_seq = np.arange(1, (len(dataset.items)+1))[
                    #     sorted_score_index]
                    # predicted_seq = items_to_predict.detach().cpu().numpy()[
                    #     sorted_score_index]
                    target_seq = session_item[1:].detach().cpu().numpy()
                # l.append((np.isin(np.unique(target_seq), np.unique(
                #     predicted_seq)).sum()/len(np.unique(target_seq))))
                # l.extend(items_to_boost[target_seq-1])
                # assert (np.isin(target_seq, np.unique(
                #     predicted_seq)).sum()/len(target_seq)) == 1
                # rank = np.nonzero(predicted_seq == target_seq[:, None])[1]
                # print(rank) # todo ground label rank
                if idx % 2 == 0:
                    tester_val.evaluate_sequence(
                        predicted_seq, target_seq, len(target_seq))
                    _, _, val_ndcg = tester_val.get_stats()
                    val_ndcg_list.append(val_ndcg)
                    coco_ndcg_list.append(val_ndcg)
                else:
                    tester_tst.evaluate_sequence(
                        predicted_seq, target_seq, len(target_seq))
                    _, _, test_ndcg = tester_tst.get_stats()
                    test_ndcg_list.append(test_ndcg)
                    coco_ndcg_list.append(test_ndcg)



            # 执行t检验
            # p_value_val, sig_val = t_test(val_ndcg_list, coco_ndcg_list)
            # p_value_test, sig_test = t_test(test_ndcg_list, coco_ndcg_list)
            #
            # # 输出结果到日志文件
            # print(f"Val集NDCG@20差异p-value: {p_value_val:.4f} ({sig_val})", file=log_file, flush=True)
            # print(f"Test集NDCG@20差异p-value: {p_value_test:.4f} ({sig_test})", file=log_file, flush=True)

            score_message, recall5, recall20 = tester_val.get_stats()
            # print(score_message.shape)
            # print("5:", recall5)
            # print("20:", recall20)
            # np.savetxt('a.txt', np.array(l)[:, None])
            # print(np.mean(l))
            print(score_message)
            print(score_message, file=log_file, flush=True)
            if recall20 > best_recall20:
                patience_idx = 0
                print('better recall@20!')
                print(f'better recall@20!', file=log_file, flush=True)
                best_recall20 = recall20
                best_test, _, _ = tester_tst.get_stats()
                print('Result on test set:')
                print(best_test, file=log_file, flush=True)
                print(best_test)
            else:
                patience_idx += 1
                print(f'patience {patience_idx}|{args.patience}')
        if patience_idx >= args.patience:
            print(f'early stop!')
            break

    print('train finished. \nbest performence:')
    print(f'train finished.', file=log_file, flush=True)
    print(best_test)
    print(best_test, file=log_file, flush=True)
    torch.save(model.state_dict(), args.save_path)
    print('result saved in: ', PRJ_PATH/'results' /
          f'my-{args.data}-ss{args.sample_size}-c{args.c:.1f}-b{args.b1:.1f}-{args.d}.txt')
    log_file.close()


def mask_no_recent(user_ids, sess_item, rcnt_item, all_item):
    '''
    过滤掉recnet_item为空的sample
    '''
    mask = rcnt_item.sum(1) != 0
    if mask.sum() == len(mask):
        return user_ids, sess_item, rcnt_item, all_item
    # if mask.sum() == 0:
    #     print('no sample has recent items')
    user_ids_ = user_ids[mask]
    sess_item_ = sess_item[mask]
    sess_len = (sess_item_ != 0).sum(1).max()
    sess_item_ = sess_item_[:, -sess_len:]
    rcnt_item_ = rcnt_item[mask]
    all_item_ = [d for d, m in zip(all_item, mask) if m]

    return user_ids_, sess_item_, rcnt_item_, all_item_


class Tester:
    def __init__(self, session_length=3, k_list=[5, 10, 20]):
        self.k_list = k_list
        self.session_length = session_length
        self.n_decimals = 4
        self.initialize()

    def initialize(self):
        self.i_count = np.zeros(self.session_length)  # [0]*self.session_length
        # [[0]*len(self.k) for i in range(self.session_length)]
        self.recall = np.zeros((self.session_length, len(self.k_list)))
        # [[0]*len(self.k) for i in range(self.session_length)]
        self.mrr = np.zeros((self.session_length, len(self.k_list)))
        self.ndcg = np.zeros((self.session_length, len(self.k_list)))

    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]:
                return i+1

        raise Exception("could not find target in sequence")

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(min(self.session_length, seq_len)):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]

            for j in range(len(self.k_list)):
                k = self.k_list[j]
                if target_item in k_predictions[:k]:
                    self.recall[i][j] += 1
                    rank = self.get_rank(target_item, k_predictions[:k])
                    self.mrr[i][j] += 1.0/rank
                    self.ndcg[i][j] += 1 / math.log(rank + 1, 2)
            self.i_count[i] += 1



    def evaluate_batch(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(
                predicted_sequence, target_sequence, sequence_lengths[batch_index])

    def get_stats(self):
        score_message = "Position\tR@5   \tMRR@5 \tNDCG@5\tR@10   \tMRR@10\tNDCG@10\tR@20  \tMRR@20\tNDCG@20\n"
        current_recall = np.zeros(len(self.k_list))
        current_mrr = np.zeros(len(self.k_list))
        current_ndcg = np.zeros(len(self.k_list))
        current_count = 0
        recall_k = np.zeros(len(self.k_list))
        for i in range(self.session_length):
            score_message += "\ni<="+str(i+2)+"    \t"
            current_count += self.i_count[i]
            for j in range(len(self.k_list)):
                current_recall[j] += self.recall[i][j]
                current_mrr[j] += self.mrr[i][j]
                current_ndcg[j] += self.ndcg[i][j]

                r = current_recall[j]/current_count
                m = current_mrr[j]/current_count
                n = current_ndcg[j]/current_count

                score_message += str(round(r, self.n_decimals))+'\t'
                score_message += str(round(m, self.n_decimals))+'\t'
                score_message += str(round(n, self.n_decimals))+'\t'

                recall_k[j] = r

        recall5 = recall_k[0]
        recall20 = recall_k[2]

        return score_message, recall5, recall20




if __name__ == '__main__':
    main()
