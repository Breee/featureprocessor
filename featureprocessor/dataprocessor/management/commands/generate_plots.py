from django.core.management.base import BaseCommand
from django.db.models import Count

from dataprocessor.models import SMTFeature
from django.db.models import Q

import seaborn as sns
import pandas as pd
import numpy as np
import json


class Command(BaseCommand):
    help = 'read csv and create SMTfeatures.'

    def add_arguments(self, parser):
        # Positional arguments are standalone name
        pass

    def reject_outliers(self, data, m=2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.)
        return data[s < m]

    def aggregate_dupes(self, frames):
        print("collecting duplicates")
        duplicates = SMTFeature.objects.values('assertion_stack_hashcode').annotate(
                duplicate_count=Count('assertion_stack_hashcode')).filter(duplicate_count__gt=1)
        dupe_hashs = []
        print(f"found {duplicates.__len__()} which occur more than one")
        for dupe in duplicates:
            dupe_hash = dupe['assertion_stack_hashcode']
            dupe_hashs.append(dupe_hash)
            feature_stack = SMTFeature.objects.filter(assertion_stack_hashcode=dupe_hash)
            x = feature_stack[0]
            solver_times = np.array([k.solver_time for k in feature_stack])
            solver_times_without_outliers = self.reject_outliers(solver_times)
            merged_frame_columns = [[x.number_of_variables,
                                     x.number_of_quantifiers,
                                     x.number_of_functions,
                                     x.biggest_equivalence_class,
                                     x.dependency_score,
                                     np.mean(solver_times_without_outliers)]]
            frames.append(pd.DataFrame(merged_frame_columns,
                                       columns=['num_vars', 'num_quantifier', 'num_funcs', 'eqiv', 'dependency',
                                                'solvertime']))

    def collect_sorts_functions_quantifiers_solvers(self):
        features = SMTFeature.objects.all()
        sorts = []
        functions = []
        quantifiers = []
        solvers = []
        results = []
        for x in features:
            for k,v in x.occuring_sorts.items():
                sorts.append(k)
            for k, v in x.occuring_functions.items():
                functions.append(k)
            for k, v in x.occuring_quantifiers.items():
                quantifiers.append(k)
            solvers.append(x.solver_name)
            results.append(x.solver_result)
        return sorted(set(sorts)), sorted(set(functions)), sorted(set(quantifiers)), sorted(set(solvers)), sorted(set(results))

    def generate_plot(self,df, xcolumns, ycolumns, solver, name, lower, upper):
        plot = sns.pairplot(data=df, hue='solver_result', y_vars=ycolumns, x_vars=xcolumns)
        plot.set(ylim=(lower, upper))
        plot.fig.suptitle(f'{solver}: range from {lower} to {upper} milliseconds solver_time', y=1.1)
        plot.savefig(f"{solver}_{lower}_to_{upper}_{name}.png")

    def handle(self, *args, **kwargs):
        frames = []
        sorts, functions, quantifiers, solvers, results = self.collect_sorts_functions_quantifiers_solvers()
        col_names = ['number_of_functions',
                     'number_of_quantifiers',
                     'number_of_variables',
                     'number_of_arrays',
                     'dagsize',
                     'treesize',
                     'dependency_score',
                     'variable_equivalence_class_sizes',
                     'biggest_equivalence_class',
                     ] + sorts + functions + quantifiers \
                    + ['contains_arrays',
                       'assertion_stack',
                       'assertion_stack_hashcode',
                       'solver_result',
                       'solver_time',
                       'solver_name']

        #print("Generating plots")
        #plot1 = sns.pairplot(df)
        #plot1.savefig("all.png")
        #
        for solver in solvers:
            for lower,upper in [(0,1000),(0,5000), (0,50000), (0,100000), (100000,200000), (200000, 500000), (0,500000), (0,1000000)]:
                print("generating dataframe list")
                dataframes = []
                for x in SMTFeature.objects.filter(solver_name=solver).filter(solver_time__lt=upper).filter(solver_time__gt=lower):
                    cols = [x.number_of_functions,
                            x.number_of_quantifiers,
                            x.number_of_variables,
                            x.number_of_arrays,
                            x.dagsize,
                            x.treesize,
                            x.dependency_score,
                            x.variable_equivalence_class_sizes,
                            x.biggest_equivalence_class] \
                           + [0 if key not in x.occuring_sorts else x.occuring_sorts[key] for key in sorts] \
                           + [0 if key not in x.occuring_functions else x.occuring_functions[key] for key in functions] \
                           + [0 if key not in x.occuring_quantifiers else x.occuring_quantifiers[key] for key in
                              quantifiers] \
                           + [1 if x.contains_arrays else 0,
                              x.assertion_stack,
                              x.assertion_stack_hashcode,
                              x.solver_result,
                              x.solver_time,
                              x.solver_name]
                    dataframe = pd.DataFrame([cols], columns=col_names)
                    dataframes.append(dataframe)
                frames = frames + dataframes
                print("concatenate frames")
                df = pd.concat(frames)
                print(f"generating plots for {solver},  {lower} to {upper} ms")

                plot_columns = ['number_of_functions',
                                'number_of_variables',
                                'number_of_arrays',
                                'treesize',
                                'dependency_score',
                                'biggest_equivalence_class']
                self.generate_plot(df=df, xcolumns=plot_columns, ycolumns=['solver_time'], solver=solver, name='metrics', lower=lower,upper=upper)


                plot_columns = ['(Array Int (Array Int Int))',
                                '(Array Int Bool)',
                                '(Array Int Int)',
                                'Bool',
                                'Int',
                                'Real']
                self.generate_plot(df=df, xcolumns=plot_columns, ycolumns=['solver_time'], solver=solver, name='sorts', lower=lower,upper=upper)

                plot_columns = ['*',
                                '+',
                                '-',
                                '/',
                                '<',
                                '<=',
                                '=',
                                '=>',
                                '>=',
                                'Positive_RA_Alt_Thresh',
                                'and',
                                'div',
                                'ite',
                                'mod',
                                'not',
                                'or',
                                'select',
                                'store']
                self.generate_plot(df=df, xcolumns=plot_columns, ycolumns=['solver_time'], solver=solver, name='functions', lower=lower,upper=upper)







