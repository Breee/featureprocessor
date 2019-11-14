from django.core.management.base import BaseCommand
from django.db.models import Count

from dataprocessor.models import SMTFeature
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import autosklearn.regression
import sklearn.metrics


class Command(BaseCommand):
    help = 'read csv and create SMTfeatures.'

    def add_arguments(self, parser):
        pass

    def reject_outliers(self, data, m=2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.)
        return data[s < m]

    def aggregate_duplicates(self, frames):
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
                                     x.number_of_functions,
                                     x.number_of_arrays,
                                     x.contains_arrays,
                                     x.occuring_functions,
                                     x.occuring_sorts,
                                     x.dagsize,
                                     x.biggest_equivalence_class,
                                     x.dependency_score,
                                     np.mean(solver_times_without_outliers)]]
            frames.append(pd.DataFrame(merged_frame_columns,
                                       columns=['number_of_variables',
                                                'number_of_functions',
                                                'number_of_arrays',
                                                'contains_arrays',
                                                'occuring_functions',
                                                'occuring_sorts',
                                                'dagsize',
                                                'biggest_equivalence_class',
                                                'dependency_score',
                                                'solvertime']))

        return dupe_hashs

    def handle(self, *args, **kwargs):
        frames = []
        dupe_hashs = self.aggregate_duplicates(frames)
        print("merging dataframe list")
        frames = frames + [pd.DataFrame([[x.number_of_variables,
                                          x.number_of_functions,
                                          x.number_of_arrays,
                                          x.contains_arrays,
                                          x.occuring_functions,
                                          x.occuring_sorts,
                                          x.dagsize,
                                          x.biggest_equivalence_class,
                                          x.dependency_score,
                                          x.solver_time]], columns=['number_of_variables',
                                                                    'number_of_functions',
                                                                    'number_of_arrays',
                                                                    'contains_arrays',
                                                                    'occuring_functions',
                                                                    'occuring_sorts',
                                                                    'dagsize',
                                                                    'biggest_equivalence_class',
                                                                    'dependency_score',
                                                                    'solvertime'])
                           for x in SMTFeature.objects.all()]
        df = pd.concat(frames)
        df['contains_arrays'] = df['contains_arrays'].replace({False: 0, True: 1})
        df = df.assign(**{function: [1 if function in cell else 0 for cell in df.occuring_functions.tolist()]
                              for function in set(function for functions in df.occuring_functions.tolist()
                                                  for function in functions)})
        df = df.assign(**{function: [1 if function in cell else 0 for cell in df.occuring_sorts.tolist()]
                          for function in set(function for functions in df.occuring_sorts.tolist()
                                              for function in functions)})
        df = df.drop('occuring_functions', axis=1)
        df = df.drop('occuring_sorts', axis=1)

        # print("Generating plots")
        # plot = sns.pairplot(df[(df['solvertime'] <= 200000) & (df['num_funcs'] <= 5000) & (df['num_vars'] <= 5000) & (
        #            df['eqiv'] <= 5000)])
        # plot.savefig("output3.png")

        X = df.drop('solvertime', axis=1)
        y = df['solvertime']

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=3600,
                per_run_time_limit=300,
                tmp_folder='/tmp/autosklearn_regression_test_tmp',
                output_folder='/tmp/autosklearn_regression_test_out',
        )
        automl.fit(X_train, y_train, dataset_name='SMT')
        # This call to fit_ensemble uses all models trained in the previous call
        # to fit to build an ensemble which can be used with automl.predict()
        automl.fit_ensemble(y_train, ensemble_size=50)

        print(automl.show_models())
        predictions = automl.predict(X_test)
        print(automl.sprint_statistics())
        print("Accuracy score", sklearn.metrics.r2_score(y_test, predictions))

