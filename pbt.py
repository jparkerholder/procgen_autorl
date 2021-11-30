import copy
from copy import deepcopy
from utils import get_base_config, get_random_config, convert_to_config, convert_to_vec
from search_space import get_hparams
import random
import numpy as np
import pandas as pd
import GPy
import shutil
from pb2_utils import normalize, optimize_acq, \
    select_length, UCB, standardize, TV_SquaredExp, TV_MixtureViaSumAndProduct
from exp3 import exp3_get_cat

class PBT(object):

    def __init__(self, args):

        self.resample_prob = args.pbt_resample
        self.mutations = get_hparams(args)
        self.perturb_amount = [0.8, 1.2]
        self.categorical_prob = 0.5
        self.method = args.search
        self.fixed_cat_val = args.fixed_cat_val
        self.t_criteria = args.budget_type
        self.budget_type = args.budget_type
        self.cat_exp = args.cat_exp
        self.numRounds = int(args.max_budget / args.t_ready)
        self.running = {}

    def exploit(self, args, agent, df, pop):
        
        if self.method == 'random':
            pop[agent]['config'] = get_random_config(args, agent=0, init=0)
            return pop[agent], 0
        
        eps = 0

        if df[df['Agent'] == agent].t.empty:
            return pop[agent]
        else:
            n = max(int(args.batchsize * args.pbt_thresh), 1)
            max_t = df.t.max()  # last iteration entry
            last_entries = df[df['t'] == max_t]  # index entire population based on last set of runs
            last_entries = last_entries.iloc[:args.batchsize] ## only want the original entries
            ranked_last_entries = last_entries.sort_values(by=['R'], ignore_index=True)  # rank last entries
            position = list(ranked_last_entries.Agent.values).index(agent) + 1  # not indexed to zero
            if position <= n:
                best_agents = list(ranked_last_entries.iloc[-n:]['Agent'].values)
                best_agent = random.sample(best_agents, 1)[0]
                best_path = '../pb2_checkpoints/' + pop[best_agent]['path']
                current_path = '../pb2_checkpoints/' + pop[agent]['path']
                shutil.copy(best_path, current_path)
                
                new_config, eps = self.explore(args, agent, best_agent, pop[best_agent]['config'], df)
                pop[agent]['config'] = new_config
                if self.cat_exp ==  'cocabo':
                    pop[agent]['Eps_cont'] = eps[0]
                    pop[agent]['Eps_cat'] = eps[1]
                else:
                    pop[agent]['Eps_cont'] = eps
                    pop[agent]['Eps_cat'] = 0
                
                print("\n replaced agent {} with agent {}".format(agent, best_agent))
                print(pop[agent]['config'])
            else:
                # not exploiting, not exploring... move on :)
                best_agent = copy.copy(agent)

        return pop[agent], best_agent

    def explore(self, args, agent, best_agent, config, df):

        if self.method == 'PBT':
            eps = 0
            new_config = self.explore_PBT(args, config)
            return new_config, eps

        elif self.method == 'PB2':
            return self.explore_PB2(args, agent, best_agent, df)


    def explore_PBT(self, args, config):

        print("\nPBT Explore\n")

        to_use = []
        current = convert_to_vec(args, config)
        for i in range(len(self.mutations)):
            row = self.mutations.iloc[i]
            if row.Type == 'continuous':
                new_val = config[row.Name] * self.perturb_amount[round(np.random.rand())]
                new_val = np.clip(new_val, row.Range[0], row.Range[1])
                to_use.append(new_val)
            elif row.Type == 'categorical':
                if self.cat_exp == 'fixed':
                    to_use.append(self.fixed_cat_val)
                else:
                    if np.random.rand() > self.categorical_prob:
                        to_use.append(row.Range[round(np.random.uniform() * (len(row.Range) - 1))])
                    else:
                        to_use.append(current[i])

        df_hparams = self.mutations.copy()
        df_hparams['Use'] = to_use

        new_config = convert_to_config(args, df_hparams)

        return new_config

    def format_df(self, args, agent, copied, df):
        """
        Helper func for PB2 methods. 

        Input: args, the agent index, and total df
        Output: dfnewpoint: New fixed params, data: formatted data
        """

        ## Get current
        n = max(int(args.batchsize * args.pbt_thresh), 1)
        agent_t = df[df['Agent'] == agent].t.max()  # last iteration entry
        last_entries = df[df['t'] == agent_t]  # index entire population based on last set of runs
        ranked_last_entries = last_entries.sort_values(by=['R'], ignore_index=True)  # rank last entries
        best_agents = list(ranked_last_entries.iloc[-n:]['Agent'].values)

        not_exploring = list(ranked_last_entries.iloc[:-n]['Agent'].values)
        for a in not_exploring:
            try:
                self.running[str(agent_t)].update(
                    {str(a): df[(df['Agent'] == a) & (df['t'] == agent_t)]['conf'].values[0]})
            except KeyError:
                self.running.update(
                    {str(agent_t): {str(a): df[(df['Agent'] == a) & (df['t'] == agent_t)]['conf'].values[0]}})

        data = df[['Agent', 't', self.budget_type, 'R']]
        data[['x{}'.format(i) for i in range(len(df.conf[0]))]] = pd.DataFrame(df.conf.tolist(), index=df.index)

        data["y"] = data.groupby(["Agent"] + ['x{}'.format(i) for i in range(len(df.conf[0]))])["R"].diff()
        data["t_change"] = data.groupby(["Agent"] + ['x{}'.format(i) for i in range(len(df.conf[0]))])[
            self.budget_type].diff()
            
        data = data[data["t_change"] > 0].reset_index(drop=True)
        data["R_before"] = data.R - data.y
            
        data["y"] = data.y / data.t_change
        data = data[~data.y.isna()].reset_index(drop=True)
        data = data.sort_values(by=self.budget_type).reset_index(drop=True)
        data = data.iloc[-1000:, :].reset_index(drop=True)
        dfnewpoint = data[data["Agent"] == copied]
        return dfnewpoint, data, agent_t

    def explore_PB2(self, args, agent, copied, df):
        
        print("\nPB2 Explore\n")

        self.cont_vars = ['x{}'.format(i) for idx, i in enumerate(range(len(df.conf[0]))) if self.mutations.Type.values[idx]=='continuous']
        self.cat_vars = ['x{}'.format(i) for idx, i in enumerate(range(len(df.conf[0]))) if self.mutations.Type.values[idx]=='categorical']
        self.all_vars = ['x{}'.format(i) for i in range(len(df.conf[0]))]
        
        dfnewpoint, data, agent_t = self.format_df(args, agent, copied, df)
        
        if not dfnewpoint.empty:
            
            to_use = {'x{}'.format(i):0 for i in range(len(self.mutations))}       
            
            ## select categorical variables first
            for i in range(len(self.mutations)):
                row = self.mutations.iloc[i]
                
                if row.Type == 'categorical':
                    
                    if self.cat_exp == 'fixed':
                        to_use['x{}'.format(i)] = self.fixed_cat_val
                    elif self.cat_exp == 'random':
                        # PB2-Rand
                        if np.random.rand() > self.categorical_prob:
                            to_use['x{}'.format(i)] = row.Range[round(np.random.uniform() * (len(row.Range) - 1))]
                        else:
                            to_use['x{}'.format(i)] = df[df['Agent'] == copied].iloc[-1].conf[i]
                    elif self.cat_exp in ['exp3_indep', 'exp3_dep', 'cocabo']:
                        # PB2-Adv/PB2-CoCa
                        data_cat = data.copy()
                        data_cat["y_exp3"] = normalize(data_cat['y'], data_cat['y'])
                        pendingactions = [x[i] for x in self.running[str(agent_t)].values()]
                        cat = exp3_get_cat(row, data_cat, self.numRounds, pendingactions)
                        to_use['x{}'.format(i)] = cat                    
            
            y = np.array(data.y.values)
            t_r = data[[self.budget_type, "R_before"]]      
            
            # choose data for the model
            if self.cat_exp in ['random', 'exp3_indep', 'exp3_dep']:
                hparams = data[self.cont_vars]
                current = [[x for idx, x in enumerate(curr) if self.mutations.Type.values[idx]=='continuous'] for curr in self.running[str(agent_t)].values()]
            elif self.cat_exp in ['cocabo']:
                hparams = data[self.all_vars]
                current = [x for x in self.running[str(agent_t)].values()]
                
            X = pd.concat([t_r, hparams], axis=1).values
                
            # exp3_dep: subset the data for separate GPs. This is why it is 'dep' :)
            if self.cat_exp == 'exp3_dep':
                rows_keep = set([x for x in range(len(data))])
                for i in range(len(self.mutations)):
                    row = self.mutations.iloc[i]
                    if row.Type == 'categorical':
                        rows_keep = rows_keep.intersection(set(data.index[data['x{}'.format(i)] == to_use['x{}'.format(i)]].tolist()))
                        
                # filter based on rows with same cats
                X = X[[x for x in rows_keep], :]
                y = y[[x for x in rows_keep]]   
                
                if X.shape[0] == 0:
                    to_use = df[df['Agent'] == copied].iloc[-1].conf
                    eps = 0
                else:
                    newpoint = dfnewpoint.iloc[-1, :][[self.budget_type, "R_before"]].values
                    new, eps = self.select_config(X, y, current, newpoint, self.mutations, num_f=len(t_r.columns))
                    for i, cont_idx in enumerate(self.cont_vars):
                        to_use[cont_idx] = new[i]
                
                    to_use = list(to_use.values())
            elif self.cat_exp == 'cocabo':
                newpoint = dfnewpoint.iloc[-1, :][[self.budget_type, "R_before"] + self.cat_vars].values
                new, eps = self.select_config(X, y, current, newpoint, self.mutations, num_f=len(t_r.columns))
                for i, cont_idx in enumerate(self.cont_vars):
                    to_use[cont_idx] = new[i]
                to_use = list(to_use.values())
            else:
                newpoint = dfnewpoint.iloc[-1, :][[self.budget_type, "R_before"]].values
                new, eps = self.select_config(X, y, current, newpoint, self.mutations, num_f=len(t_r.columns))
                for i, cont_idx in enumerate(self.cont_vars):
                    to_use[cont_idx] = new[i]
                to_use = list(to_use.values())
        else:
            random_config = get_random_config(args)
            to_use = [random_config[x] for x in self.mutations.Name.values]
            if self.cat_exp=='cocabo':
                eps = [0,0]
            else:
                eps = 0

        try:
            self.running[str(agent_t)].update({str(agent): to_use})
        except KeyError:
            self.running.update({str(agent_t): {str(agent): to_use}})

        df_hparams = self.mutations.copy()
        df_hparams['Use'] = to_use

        new_config = convert_to_config(args, df_hparams)

        return new_config, eps


    def select_config(self, Xraw, yraw, current, newpoint, mutations, num_f):
        """Selects the next hyperparameter config to try.
        """
            
        oldpoints = Xraw[:, :num_f]
        X_use = Xraw[:, num_f:]
        
        if self.cat_exp == 'cocabo':
            cat_dims = [int(x[1]) for x in self.cat_vars]
            X_cat = X_use[:, cat_dims]
            X_use = X_use[:, [x for x in range(X_use.shape[1]) if x not in cat_dims]]
            fixed_cat = newpoint[num_f:]
            newpoint = newpoint[:num_f]
            current_cat = [[x for idx, x in enumerate(curr) if idx in cat_dims] for curr in current]
            current = [[x for idx, x in enumerate(curr) if idx not in cat_dims] for curr in current]
        else:
            cat_dims = []
        
        X_use = np.concatenate((oldpoints, X_use), axis=1)
            
        base_vals = [val for (val, x) in zip(mutations.Range.values, mutations.Type.values) if x is not 'categorical']
        base_vals = np.array(base_vals).T[::-1]
        
        fixed_points = np.concatenate((oldpoints, newpoint.reshape(1,-1)), axis=0)
        old_lims = np.concatenate((np.max(fixed_points, axis=0),
                                   np.min(fixed_points, axis=0))).reshape(2, oldpoints.shape[1])
    
        old_lims[0] -= 1e-8
        old_lims[1] += 1e-8        
    
        limits = np.concatenate((old_lims, base_vals), axis=1)
        limits[0] -= 1e-8
        limits[1] += 1e-8
        
        X = normalize(X_use, limits)
        y = standardize(yraw).reshape(yraw.size, 1)
    
        fixed = normalize(newpoint, old_lims)
        
        if self.cat_exp in ['random', 'exp3_indep', 'exp3_dep']:
            
            current = [[x for idx, x in enumerate(entry) if idx not in cat_dims] for entry in current]
            kernel = TV_SquaredExp(
                input_dim=X.shape[1], variance=1., lengthscale=1., epsilon=0.1)
            
        elif self.cat_exp == 'cocabo':
            
            X = np.concatenate((X[:, :num_f], X_cat, X[:, num_f:]), axis=1)
            
            cat_locs = [x+num_f for x in range(X_cat.shape[1])]
            
            kernel = TV_MixtureViaSumAndProduct(X.shape[1],
                 variance_1=1.,
                 variance_2=1.,
                 variance_mix=1.,
                 lengthscale=1.,
                 epsilon_1=0.,
                 epsilon_2=0.,
                 mix = 0.5,
                 cat_dims = cat_locs)
    
        try:
            m = GPy.models.GPRegression(X, y, kernel)
        except np.linalg.LinAlgError:
            # add diagonal ** we would ideally make this something more robust...
            X += np.eye(X.shape[0]) * 1e-3
            m = GPy.models.GPRegression(X, y, kernel)
    
        try:
            m.optimize()
        except np.linalg.LinAlgError:
            # add diagonal ** we would ideally make this something more robust...
            X += np.eye(X.shape[0]) * 1e-3
            m = GPy.models.GPRegression(X, y, kernel)
            m.optimize()
        
        if self.cat_exp in ['random', 'exp3_indep', 'exp3_dep']:            
            m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5, 1))
        elif self.cat_exp == 'cocabo':
            m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5, 1))
    
        if current is None:
            m1 = deepcopy(m)
        else:
            # add the current trials to the dataset
            current_use = normalize(current, base_vals)
            padding = np.array([fixed for _ in range(current_use.shape[0])])
            
            if self.cat_exp == 'cocabo':
                current_use = np.concatenate((padding, current_cat, current_use), axis=1)
            else:
                current_use = np.hstack((padding, current_use))
    
            Xnew = np.vstack((X, current_use))
            
            # y value doesn't matter, only care about the variance.
            ypad = np.zeros(current_use.shape[0])
            ypad = ypad.reshape(-1, 1)
            ynew = np.vstack((y, ypad))
    
            if self.cat_exp in ['random', 'exp3_indep', 'exp3_dep']:
                
                cat_dims = []
                kernel = TV_SquaredExp(
                    input_dim=X.shape[1], variance=1., lengthscale=1., epsilon=0.1)
                
            elif self.cat_exp == 'cocabo':
                
                cat_dims = [int(x[1])-1 for x in self.cat_vars]
                
                kernel = TV_MixtureViaSumAndProduct(Xnew.shape[1],
                     variance_1=1.,
                     variance_2=1.,
                     variance_mix=1.,
                     lengthscale=1.,
                     epsilon_1=0.,
                     epsilon_2=0.,
                     mix = 0.5,
                     cat_dims = cat_locs)
                
            m1 = GPy.models.GPRegression(Xnew, ynew, kernel)
            m1.optimize()
        
        if self.cat_exp == 'cocabo':
            fixed = np.concatenate((fixed.reshape(1,-1), fixed_cat.reshape(1,-1)), axis=1)
            xt = optimize_acq(UCB, m, m1, fixed, num_f + len(cat_dims))
        else:
            xt = optimize_acq(UCB, m, m1, fixed, num_f)
    
        # convert back...
        xt = xt * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + np.min(
            base_vals, axis=0)
    
        xt = xt.astype(np.float32)
        
        epsilon = 0
        if self.cat_exp in ['random', 'exp3_indep', 'exp3_dep']:
            epsilon = m.kern.epsilon[0]
        elif self.cat_exp == 'cocabo':
            epsilon = [m.kern.epsilon_1[0], m.kern.epsilon_2[0]]
        
        return (xt, epsilon)

