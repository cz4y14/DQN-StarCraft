#!/usr/lib/python2.7
# --*--coding:UTF-8--*--
import argparse
import sys
sys.path.append("game/")
#sys.path.append("/home/ysh/gym-flay-starcrat/gym-starcraft")
import gym_starcraft.envs.single_battle_env as sc

import time
import cv2


#import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np
import random


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        #self.action_rel=[0,0,0]
        self.action_rel = [0, 0, 0, 0]
    def act(self):
        #print "1"
        #return self.action_space.sample()
        self.action_rel= self.action_space.sample()
        return self.action_rel
        #action_index=random.randint(0,3)
        #if self.action_rel[0] > 0:
        #    self.action_rel[0]=1.0
        #    self.action_rel[1] = 0
        #    self.action_rel[2] = 0
        #else:
        #    self.action_rel[0]=-1.0
        #    self.action_rel[1]=action_index*90
        #    self.action_rel[2]=16
        #return self.action_rel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='server ip')
    parser.add_argument('--port', help='server port', default="11111")
    args = parser.parse_args()

   
    env = sc.SingleBattleEnv(args.ip, args.port)
    env.seed(123)
    agent = RandomAgent(env.action_space)

    episodes = 0
    brain = BrainDQN(7)
    p_state = {}
    obs,p_state = env.reset()
    action = agent.act()
    brain.setInitState(obs)
    #因为reset的延时问题导致刚开始的几帧画面没有units，所以此处加上异常处理机制
    #try:
    while((len(p_state['units_myself']) == 0) or (len(p_state['units_enemy']) == 0)):
          #print "p_state",p_state
          obs, p_state = env.reset()
          action = agent.act()
          brain.setInitState(obs)
    #finally:
    #    print "the frame of StarCraft is going on"

    action_pl = brain.getAction(action, episodes,p_state)
    #unit_myself_index = random.randint(0, 2)
    unit_myself_index = random.randint(0,len(p_state['units_myself'])-1)
    obs, reward, done, info,p_state = env.step(action_pl,unit_myself_index)

    action_nerual =[0,0,0,0,0,0,0]
    startime = time.time()
    while episodes <60000:
        # print "1"
        #if episodes % 5==0:
        #    endtime=time.time()
        #    gtime=endtime-startime
        #    print("%d lun time=%s"%(episodes,gtime))
        #obs = env.reset()
        done = False
        #train
        #unit_myself_index = 0

        while  not done:
            action = agent.act()
        #在敌我双方敌人一方将死的瞬间，getAction下标出界，因此加入条件判断
            action_pl = brain.getAction(action, episodes, p_state)
        #在敌我双方其中一方将死的瞬间，done仍为false，因此加入条件判断
            if (len(p_state['units_myself'])!=0  ):
                unit_myself_index = random.randint(0, len(p_state['units_myself']) - 1)
                obs, reward, done, info, p_state = env.step(action_pl, unit_myself_index)
            else:
                unit_myself_index = 0
                obs, reward, done, info, p_state = env.step(action_pl, unit_myself_index)
                #done = True
            #obs, reward, done, info, p_state = env.step(action_pl, unit_myself_index)
            #print done
            if action_pl[0] == 1.0 and action_pl[3] == 0:
                action_nerual = [1.0, 0, 0, 0, 0, 0, 0]
            elif action_pl[0] == 1.0 and action_pl[3] == 1:
                action_nerual = [0, 1.0, 0, 0, 0, 0, 0]
            elif action_pl[0] == 1.0 and action_pl[3] == 2:
                action_nerual = [0, 0, 1.0, 0, 0, 0, 0]
            elif action_pl[0] == -1.0 and action_pl[1] == 0:
                action_nerual = [0, 0, 0, 1.0, 0, 0, 0]
            elif action_pl[0] == -1.0 and action_pl[1] == 90:
                action_nerual = [0, 0, 0, 0, 1.0, 0, 0]
            elif action_pl[0] == -1.0 and action_pl[1] == 180:
                action_nerual = [0, 0, 0, 0, 0, 1.0, 0]
            else:
                action_nerual = [0, 0, 0, 0, 0, 0, 1.0]

            brain.setPerception(obs, action_nerual, reward, done)
        #unit_myself_index += 1

            #print(action,obs,reward,done,info)
        episodes += 1
        #obs = env.reset()
        _, p_state = env.reset()
        #test
        total_reward=0
        avg_reward=0
        TEST_EPISODE=50
        if episodes % 2000 == 0:
            endtime=time.time()
            gtime=endtime-startime
            print gtime
            for test_i in xrange(TEST_EPISODE):
                unit_myself_index = 0
                obs_test,p_state= env.reset()
                done_test = False
                step_episode = 0
                while not done_test:
                    #if (step_episode / 10) % (len(p_state['units_myself'])) == 0:
                    #    unit_myself_index = 3-len(p_state['units_myself'])
                    #elif (step_episode / 10) % len(p_state['units_myself']) == 1:
                    #    unit_myself_index = 3-len(p_state['units_myself'])
                    #else:
                    action_test=brain.getActionForTest(obs_test)
                    if (len(p_state['units_myself']) != 0):
                        unit_myself_index = random.randint(0, len(p_state['units_myself']) - 1)
                        obs_test,reward_test,done_test,info_test,p_state = env.step(action_test,unit_myself_index)
                    else:
                        unit_myself_index = 0
                        obs_test, reward_test, done_test, info_test, p_state = env.step(action_test, unit_myself_index)
                    #print action_test,reward_test
                    total_reward+=reward_test

            avg_reward=total_reward/TEST_EPISODE
            print "经过",episodes, "轮","耗时",gtime,"测试",TEST_EPISODE,"局","总奖励",total_reward,"平均奖励",avg_reward
            #将信息输出至文本
            with open("outputIF.txt","a+") as f:
                print>>f,"经过", episodes, "轮", "耗时", gtime, "测试", TEST_EPISODE, "局", "总奖励", total_reward, "平均奖励", avg_reward
        if avg_reward>5000:
            break

    env.close()
