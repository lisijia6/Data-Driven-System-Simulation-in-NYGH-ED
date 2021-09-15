import pandas as pd
from resource import *
import heapq as hq
import numpy as np
import os

class multi_class_single_station_priority:
    # defining the queueing system using given parameters
    def __init__(self, **kwargs):

        # initialize parameters
        self.lambda_ = kwargs.get('lambda_', 1)  # arrival rate
        self.classes_ = kwargs.get('classes', [0])  # class types

        self.probs = kwargs.get('probs', [1])  # probability of arrival for each class

        self.mus = kwargs.get('mus',[1])  # service rate without intervention
        self.probs_speedup = kwargs.get('prob_speedup', [0]*len(self.classes_))  # probability of speedup

        self.mus_speedup = kwargs.get('mus_speedup', self.mus)  # service rate with intervention
        self.servers = kwargs.get('servers', 1)  # number of servers

        self.priority = kwargs.get('priority', [0])  # priority assignment, smaller number means higher priority

        # initialize trackers with relevant statistics, assume all start empty
        self.data = []  # event logs
        self.wait_time_tracker = []  # [[[timestamp, timestamp, ...], ...]]
        self.los_tracker = []  # [[[timestamp, timestamp, ...], ...]]
        self.queue_tracker = []  # [[[(timestamp, Nq), (timestamp, Nq), ...], ...]]
        self.nis_tracker = []  # [[[(timestamp, NIS), (timestamp, NIS), ...], ...]]


    # Getters
    def get_classes(self):
        return self.classes_

    def get_wait_time_tracker(self):
        return self.wait_time_tracker

    def get_los_tracker(self):
        return self.los_tracker

    def get_queue_tracker(self):
        return self.queue_tracker

    def get_nis_tracker(self):
        return self.nis_tracker


    def simulate_priority_q(self, customers, runs):

        np.random.seed(3)  # set random seed

        for r in range(runs):
            print('running simulation run #: ' + str(r + 1))

            t_ = 0  # time starts from zero
            sim_arrival_times = []
            service_times = []
            classes_ = []
            interv_ = []

            for c in range(customers):
                # simulate next arrival
                # next arrival time: t_ + inter-arrival_time
                sim_arrival_times.append(
                    t_ + Distribution(dist_type=DistributionType.exponential, rate=self.lambda_).sample())

                # move forward the timestamp
                t_ = sim_arrival_times[len(sim_arrival_times)-1]

                # sampling the class of arriving patient
                c_ = np.random.choice(self.classes_, p=self.probs)
                classes_.append(c_)

                # sampling whether intervention or not
                interv_.append(np.random.choice(np.array([0,1]),
                                                p = np.array([1-self.probs_speedup[c_], self.probs_speedup[c_]])))
                if interv_[len(interv_)-1]==0:
                    service_times.append(Distribution(dist_type=DistributionType.exponential, rate=self.mus[c_]).sample())
                else:
                    service_times.append(
                        Distribution(dist_type=DistributionType.exponential, rate=self.mus_speedup[c_]).sample())

            event_log = []
            queue_tr = [[(0, 0)] for _ in self.classes_]  # [[(timestamp, Nq), (timestamp, Nq), ...], ...]
            nis_tr = [[(0, 0)] for _ in self.classes_]  # [[(timestamp, NIS), (timestamp, NIS), ...], ...]
            los_tr = [[] for _ in self.classes_]  # [[timestamp, timestamp, ...], ...]
            wait_tr = [[] for _ in self.classes_]  # [[timestamp, timestamp, ...], ...]
            # four types of events: arrival, departure = 'a', 'd' queue and service (queue start and service start)
            # every tuple is (timestamp, event_type, customer id, server_id)
            event_calendar = [(a, 'a', i, -1, self.priority[classes_[i]]) for i, a in enumerate(sim_arrival_times)]
            hq.heapify(event_calendar)

            queue = []  # [(timestamp, customer_id), (timestamp, customer_id), ...]
            hq.heapify(queue)
            # heap is ordered by timestamp - every element is (timestamp, station)
            # need to manage server assignment
            in_service = [0 for _ in range(self.servers)]  # 0 = not in service; 1 = in service
            server_assignment = [0 for _ in range(self.servers)]
            # temp_friends = {}

            # keep going if there are still events waiting to occur
            while len(list(event_calendar))>0:
                # take an event from the event_calendar
                ts_, event_, id_, server_, priority_ = hq.heappop(event_calendar)

                # arrival event happens, need to check if servers are available
                if event_ == 'a':
                    # log arrival event
                    event_log.append((ts_, 'a', id_, interv_[id_], classes_[id_]))

                    # update nis_tracker - add 1 to the class in which the customer belongs to
                    nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] + 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    for class_ in self.classes_:
                        if classes_[id_] != class_:
                            nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))

                    # if there is a room in service
                    if sum(in_service) < self.servers:
                        for j,s in enumerate(in_service):
                            # find the first available server
                            if s == 0:
                                # set the jth server to be busy, serving customer with id_
                                in_service[j] = 1
                                server_assignment[j] = id_

                                # add a departure event to the event_calendar
                                hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,j, self.priority[classes_[id_]]))
                                break

                        # log service and departure events
                        event_log.append((ts_, 's', id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # update wait_time_tracker, wait_time = current time - arrival time
                        wait_tr[classes_[id_]].append(ts_ - sim_arrival_times[id_])
                        # update los_tracker, los = current time + service time - arrival time
                        los_tr[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        # temp_friends[id_] = []

                    # if there is no room on servers
                    else:
                        # temp_friends[id_] = [str(q[1])+"_"+str(r) for q in list(queue)]

                        # join the queue
                        hq.heappush(queue,(priority_, ts_, id_))  # smaller number means higher priority

                        # log queueing event
                        event_log.append((ts_, 'q', id_, interv_[id_], classes_[id_]))

                        # update queue_tracker - add 1 to the class in which the customer belongs to
                        queue_tr[classes_[id_]].append((ts_,queue_tr[classes_[id_]][-1][1] + 1))

                        # update queue_tracker for all other classes, Nq stays the same
                        for class_ in self.classes_:
                            if classes_[id_]!=class_:
                                queue_tr[class_].append((ts_, queue_tr[class_][-1][1]))


                # departure event happens
                else:
                    in_service[server_] = 0  # free the server
                    # server_assignment[server_] = 0
                    # update nis_tracker - subtract 1 to the class in which the customer belongs to
                    nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] - 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    for class_ in self.classes_:
                        if classes_[id_] != class_:
                            nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))

                    # if there is still customer in the queue
                    if len(list(queue)) > 0:
                        # take a customer from the queue
                        _, _ , id_ = hq.heappop(queue)

                        # log service and departure events
                        event_log.append((ts_,'s',id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # the server becomes busy again, assign a customer with id_ to the server
                        in_service[server_] = 1
                        server_assignment[server_] = id_

                        # add a departure event to the event_calendar
                        hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,server_, self.priority[classes_[id_]]))

                        # update wait_time_tracker, los_tracker, queue_tracker (subtract 1)
                        wait_tr[classes_[id_]].append(ts_ - sim_arrival_times[id_])
                        los_tr[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        queue_tr[classes_[id_]].append((ts_, queue_tr[classes_[id_]][-1][1] - 1))

                        # update the queue_tracker for all other classes, Nq stays the same
                        for class_ in self.classes_:
                            if classes_[id_]!=class_:
                                queue_tr[class_].append((ts_, queue_tr[class_][-1][1]))

            # add the event_log to "data", and append trackers for each run to overall trackers
            self.data.append(event_log)
            self.wait_time_tracker.append(wait_tr)
            self.los_tracker.append(los_tr)
            self.nis_tracker.append(nis_tr)
            self.queue_tracker.append(queue_tr)

        print('Done simulating...')


    def generate_data(self, **kwargs):
        # generating data for intervention experiments
        write_file = kwargs.get('write_file', True)

        offset = 0.0  # time at the end of last run
        directory, folder = "", ""

        # iterate through each event log (simulation run) in simulation data
        for j,e_l in enumerate(self.data):
            # print("Run #"+str(j+1))

            # creating a data-frame to manage the event logs
            # one per simulation run - we will later want to compare interventions
            df = pd.DataFrame(e_l, columns=['timestamp', 'event_type', 'id', 'A', 'C'])
            # two things: we both want plots to see if the simulator makes sense, and create synthetic data
            # print(df.head(5))
            # order by id and timestamp there may be tie between a and q - we don't care
            df.sort_values(by=['id','timestamp'], inplace=True)
            df.reset_index(drop=True,inplace=True)

            # add additional columns to the DataFrame
            df['elapsed'] = 0.0  # time elapsed since customer's arrival
            df['arrival_time'] = 0.0
            df['id_run'] = ""
            cur_id = df.at[0,'id']
            cur_start = df.at[0,'timestamp']
            # df['FriendsID'] = " "
            # df['nFriends'] = 0
            # temp_friends = self.friends[j]

            # go through each event in the DataFrame
            for i in range(len(df)):
                df.at[i,'id_run'] = str(df.at[i,'id'])+"_"+str(j)

                # if the event corresponds to the current customer
                if cur_id == df.at[i,'id']:
                    df.at[i, 'arrival_time'] = cur_start + offset
                    df.at[i,'elapsed'] = df.at[i,'timestamp'] - cur_start
                    #print(df.at[i,'event_type'])
                    #input("Press Enter to continue...")

                # if the event does not correspond to the current customer, events for the next customer starts
                else:
                    cur_id = df.at[i, 'id']  # set current customer to the customer for the event
                    cur_start = df.at[i,'timestamp'].copy()  # advance the current start time to the time of event
                    df.at[i,'arrival_time'] = cur_start + offset
                # df.at[i,'FriendsID'] = " ".join(map(str, temp_friends[df.at[i,'id']]))
                # df.at[i,'nFriends'] = len(temp_friends[df.at[i, 'id']])

            offset = offset + max(df['timestamp']) # the next simulation run starts at the offset time
            # print('Average LOS per run: ')
            # print(np.mean(df[df.event_type == 'd']['elapsed']))


            # generate csv files
            if write_file:
                # save generated data in a folder in the current working directory
                cwd = os.getcwd() # get current working directory
                # single type of customers
                if len(self.mus) == 1:
                    folder = "Priority (Non-preemptive) - Lambda{}Mu{}P(Interv){}MuPrime{}".format(self.lambda_, self.mus[0],
                                                                              self.probs_speedup[0], self.mus_speedup[0])
                # two types of customers
                elif len(self.mus) == 2:
                    folder = "Priority (Non-preemptive) - LamOne{}LamTwo{}MuOne{}MuTwo{}P(C1){}P(C1_Interv){}P(C2_Interv){}MuOnePrime{}MuTwoPrime{}".format(
                        self.lambda_, self.lambda_, self.mus[0], self.mus[1], self.probs[0], self.probs_speedup[0],
                        self.probs_speedup[1], self.mus_speedup[0], self.mus_speedup[1])
                # todo: more than 2 types of customers?

                directory = os.path.join(cwd, folder)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # generate file: 1) Queue/Waiting and System/Time

                header_ = True if j == 0 else False
                mode_ = 'w' if j == 0 else 'a'
                # file_1: Queue/Waiting and System/Time
                filename = "data_WIQ_TIS"
                save_path = os.path.join(directory, filename+".csv")
                df[df.event_type == 'd'].loc[:,
                ['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'elapsed']].to_csv(save_path,
                                                                                                   mode=mode_,
                                                                                                   index=False,
                                                                                                   header=header_)
                # wait time = service start time - arrival time
                df[df.event_type == 's'].loc[:,
                ['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'elapsed']].to_csv(save_path,
                                                                                                   mode='a',
                                                                                                   index=False,
                                                                                                   header=False)


        # generate files: 2) Queue/Number, 3) System/Number
        if write_file:
            # file_2: Queue/Number
            filename = "data_NIQ"
            save_path = os.path.join(directory, filename + ".csv")
            for r, queue_tr in enumerate(self.queue_tracker):
                df_niq = pd.DataFrame(columns=['run', 'timestamp', 'class_id', 'Number_in_Queue'])
                offset = 0
                for class_ in self.classes_:
                    for i, queue in enumerate(queue_tr[class_]):
                        df_niq.loc[i + offset] = [r+1, queue[0], class_, queue[1]]
                        offset += len(queue_tr[class_])

                df_niq.sort_values(by=['timestamp'], inplace=True)  # order by timestamp
                df_niq.reset_index(drop=True, inplace=True)

                if r == 0:
                    df_niq.to_csv(save_path, index=False, header=True)
                else:
                    df_niq.to_csv(save_path, mode='a', index=False, header=False)

            # file_3: System/Number
            filename = "data_NIS"
            save_path = os.path.join(directory, filename + ".csv")
            for r, nis_tr in enumerate(self.nis_tracker):
                df_nis = pd.DataFrame(columns=['run', 'timestamp', 'class_id', 'Number_in_System'])
                offset = 0
                for class_ in self.classes_:
                    for i, system in enumerate(nis_tr[class_]):
                        df_nis.loc[i + offset] = [r+1, system[0], class_, system[1]]
                        offset += len(nis_tr[class_])

                df_nis.sort_values(by=['timestamp'], inplace=True)  # order by timestamp
                df_nis.reset_index(drop=True, inplace=True)

                if r == 0:
                    df_nis.to_csv(save_path,index=False,header=True)
                else:
                    df_nis.to_csv(save_path, mode='a', index=False, header=False)

        # print("Average SLA value: "+str(np.mean(self.sla_levels)))


if __name__ == "__main__":
    q_priority_1 = multi_class_single_station_priority(lambda_=1, classes=[0, 1], probs=[0.5, 0.5], mus = [1.1, 1.1],
                                                       prob_speedup=[0.5, 0.5], mus_speedup=[2, 2], servers = 1,
                                                       priority=[0, 1])
    q_priority_1.simulate_priority_q(customers=100, runs=3)
    q_priority_1.generate_data(write_file=True)