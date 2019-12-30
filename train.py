import argparse
import matplotlib.pyplot as plt

from env import BaseEnv
from model.MLP import MLP
from agent import Agent
from replay_buffer import ReplayBuffer
from goal_buffer import GoalBuffer
from utils import *

parser = argparse.ArgumentParser()

# Training Options
parser.add_argument('--epoch_num', default=20000, type=int)
parser.add_argument('--max_step', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--buffer_size', default=10000, type=int)
parser.add_argument('--reg_term', default=0.5, type=float)
parser.add_argument('--grad_clip', default=10, type=float)
parser.add_argument('--random_step', default=100, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=4, type=int)
parser.add_argument('--s_a_hidden_size', default=(32, 32))
parser.add_argument('--s_hidden_size', default=(32, 32))
parser.add_argument('--s_a_lr', default=1e-4)
parser.add_argument('--s_lr', default=1e-4)

# Environment Options
parser.add_argument('--env_size', default=(10, 10))


def main(args):

    # environment initialization
    env = BaseEnv(size=args.env_size)

    # embedding network initialization
    f_s_a = MLP(env.state_size + env.action_size, args.s_a_hidden_size, args.embedding_dim)
    f_s = MLP(env.state_size, args.s_hidden_size, args.embedding_dim)

    # buffer initialization
    replay_buffer = ReplayBuffer(args, env)
    goal_buffer = GoalBuffer()
    init_goal = tuple(np.random.randint(1, args.env_size[0] + 1, size=2))
    print(init_goal)
    goal_buffer.store(init_goal)

    # agent initialization
    agent = Agent(env_size=args.env_size)

    # optimizer initialization
    s_a_optimizer = torch.optim.Adam(f_s_a.parameters(), lr=args.s_a_lr)
    s_optimizer = torch.optim.Adam(f_s.parameters(), lr=args.s_a_lr)

    log_loss = []

    for epoch in range(args.epoch_num):
        start_position = np.random.randint(1, args.env_size[0]+1, size=2)
        goal = goal_buffer.sample_batch_goal(size=1)[0]
        g_feature = agent.get_state_feature(goal)
        ns, r, terminate = env.reset(size=args.env_size, start_pos=start_position)
        for step in range(args.max_step):
            s = ns
            s_feature = agent.get_state_feature(s)
            action, min_dist = agent.get_best_action(s_feature, f_s_a, f_s, g_feature)
            ns, r, terminate = env.step(action)
            ns_feature = agent.get_state_feature(ns)
            vec_action = vectorize_action(action)
            print(s_feature.shape, vec_action.shape)
            # store one step loss
            s_a_pred = f_s_a.predict(np.concatenate((s_feature, vec_action), axis=0).reshape((1, -1)))
            ns_pred = f_s.predict(np.array(ns_feature).reshape((1, -1)))
            e = torch.norm(torch.FloatTensor(s_a_pred - ns_pred), dim=1)[0]
            replay_buffer.add((s_feature, vec_action, ns_feature, None, e))

            # store two step loss


            # print(s, s_feature, action, raw_action, ns, ns_feature, vec_action)

        s_a_batch, s_batch, g_batch = buffer.get_batch_data()
        # loss for (s, a, s')
        s_a_embed = f_s_a(s_a_batch)
        s_embed = f_s(s_batch)
        g_embed = f_s(g_batch)
        loss_normal = torch.mean(torch.norm((s_a_embed - s_embed), dim=1))
        # loss for (s, a, s', g)
        ns_best_action = np.zeros((0, 4))
        valid_list = []
        for i in range(args.batch_size):
            best_action, min_dist = agent.get_best_action(s_batch[i], f_s_a, f_s, g_batch[i])
            vec_action = vectorize_action(best_action).reshape(1, -1)
            ns_best_action = np.concatenate((ns_best_action, vec_action), axis=0)
            if all(s_embed[i] != g_embed[i]):
                valid_list.append(i)
        ns_a_batch = np.concatenate((s_batch, ns_best_action), axis=1)

        ns_a_embed = f_s_a.predict(ns_a_batch)
        g_target_embed = f_s.predict(g_batch)
        dist_s = torch.norm((s_a_embed - g_embed), dim=1)[valid_list]
        with torch.no_grad():
            # calculate target
            dist_ns = torch.norm(torch.FloatTensor(ns_a_embed - g_target_embed), dim=1)[valid_list]
            target = dist_ns + 1
        if (epoch + 1) % 100 == 0:
            print(dist_s - dist_ns)
        loss_update = torch.mean(torch.abs(dist_s - target))
        # print(loss_update)
        loss = (1-args.reg_term)*loss_normal + args.reg_term * loss_update
        log_loss.append(loss)
        s_a_optimizer.zero_grad()
        s_optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(f_s.parameters(), args.grad_clip)
        # nn.utils.clip_grad_norm_(f_s_a.parameters(), args.grad_clip)
        s_a_optimizer.step()
        s_optimizer.step()
        if (epoch + 1) % 100 == 0:
            print("epoch number: {}/{}, total_loss: {}, loss_normal: {}, loss_update: {}".format(epoch + 1,
                                                                                                args.epoch_num,
                                                                                                loss,
                                                                                                loss_normal,
                                                                                                loss_update
                                                                                                ))
        if epoch == args.epoch_num - 1:
            print(s_a_embed, s_embed)

        if (epoch + 1) % 1000 == 0 and loss < 1:
            save_model("./model.pt", f_s_a, f_s, args.epoch_num, loss, s_a_optimizer, s_optimizer)
            print("Saving model at epoch: {}".format(epoch))

        if (epoch + 1) % 10000 == 0:
            plt.plot(range(epoch + 1), log_loss, color='red')
            plt.savefig("./{}.pdf".format(int((epoch + 1)/10000)), dpi=1200, bbox_inches='tight')


    # save_model("./model.pt", f_s_a, f_s, args.epoch_num, loss, s_a_optimizer, s_optimizer)
    # print("model saved.")


        # print("Finish optimization for epoch: {}".format(epoch))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

