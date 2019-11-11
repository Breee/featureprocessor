from django.core.management.base import BaseCommand
from django.db.models import Count

from dataprocessor.models import SMTFeature
import seaborn as sns
import pandas as pd
import numpy as np

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

    def handle(self, *args, **kwargs):
        frames = []
        print("collecting duplicates")
        duplicates = SMTFeature.objects.values('assertion_stack_hashcode').annotate(duplicate_count=Count('assertion_stack_hashcode')).filter(duplicate_count__gt=1)
        dupe_hashs = []
        print(f"found {duplicates.__len__()} which occur more than one")
        for dupe in duplicates:
            dupe_hash = dupe['assertion_stack_hashcode']
            dupe_hashs.append(dupe_hash)
            feature_stack = SMTFeature.objects.filter(assertion_stack_hashcode=dupe_hash)
            x = feature_stack[0]
            #print(f"AssertionHashCode {dupe_hash} occurs {feature_stack.__len__()}")
            solver_times = np.array([k.solver_time for k in feature_stack])
            solver_times_without_outliers = self.reject_outliers(solver_times)
            #print(f"Outlier rejection shrinked size from {len(solver_times)} to {len(solver_times_without_outliers)}")
            merged_frame_columns = [[x.number_of_variables,
                             x.number_of_quantifiers,
                             x.number_of_functions,
                             x.biggest_equivalence_class,
                             x.dependency_score,
                             np.mean(solver_times_without_outliers)]]
            frames.append(pd.DataFrame(merged_frame_columns,
                                       columns=['num_vars','num_quantifier','num_funcs','eqiv','dependency','solvertime']))
        print("merging dataframe list")
        frames = frames + [pd.DataFrame([[x.number_of_variables,
                             x.number_of_quantifiers,
                             x.number_of_functions,
                             x.biggest_equivalence_class,
                             x.dependency_score,
                             x.solver_time]], columns=['num_vars','num_quantifier','num_funcs','eqiv','dependency','solvertime']) for x in SMTFeature.objects.all().exclude(assertion_stack_hashcode__in=dupe_hashs)]
        df = pd.concat(frames)

        print("Generating plots")
        plot = sns.pairplot(df[(df['solvertime'] <= 200000) & (df['num_funcs'] <= 5000) & (df['num_vars'] <= 5000) & (df['eqiv'] <= 5000)])
        plot.savefig("output2.png")


