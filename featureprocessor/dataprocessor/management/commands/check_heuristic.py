from django.core.management.base import BaseCommand
from random import randint
from itertools import chain, combinations
from dataprocessor.models import SMTFeature
from collections import namedtuple

AnalysisResult = namedtuple('AnalysisResult', 'features, correct, wrong, equal, total')

class Command(BaseCommand):
    help = 'read csv and create SMTfeatures.'

    def add_arguments(self, parser):
        parser.add_argument('dumppath', default='.', required=False)

    def smaller_wins(self,feature1, feature2 ):
        if feature1 < feature2:
            return 1,0
        elif feature1 > feature2:
            return 0,1
        return 0,0


    def predict_winner(self, feature_vector1, feature_vector2, feature_ids):
        feature1_score = 0
        feature2_score = 0
        for fid in feature_ids:
            s1, s2 = self.smaller_wins(getattr(feature_vector1, fid), getattr(feature_vector2,fid))
            feature1_score += s1
            feature2_score += s2
        if feature1_score > feature2_score:
            return True, False
        if feature1_score < feature2_score:
            return False, True
        return False, False

    def analyze_featureset(self, feature_ids, dumppath):

        print(f'=============================================================')
        print(f'Analyzing feature_ids {feature_ids}')
        print(f'=============================================================')
        correct_predictions = 0
        wrong_predictions = 0
        equal_predictions = 0
        total_predictions = 0
        test_range = 1000000

        id_list = SMTFeature.objects.values_list('id', flat=True)
        amnt = len(id_list)
        for i in range(0, test_range):
            random_object1 = SMTFeature.objects.get(id=id_list[randint(0, amnt - 1)])  # single random object
            random_object2 = SMTFeature.objects.get(id=id_list[randint(0, amnt - 1)])  # single random object
            f1, f2 = random_object1, random_object2
            if f1.id == f2.id:
                continue
            f1_wins, f2_wins = self.predict_winner(feature_vector1=f1, feature_vector2=f2, feature_ids=feature_ids)
            if f1_wins:
                if f1.solver_time < f2.solver_time:
                    correct_predictions += 1
                else:
                    wrong_predictions += 1
            elif f2_wins:
                if f2.solver_time < f1.solver_time:
                    correct_predictions += 1
                else:
                    wrong_predictions += 1
            else:
                equal_predictions += 1
            total_predictions += 1

        print(f'=============================================================')
        print(f'correct: {correct_predictions}')
        print(f'wrong: {wrong_predictions}')
        print(f'equal: {equal_predictions}')
        print(f'total: {total_predictions}')
        print(f'{(correct_predictions / total_predictions) * 100}%correct,'
              f' {(wrong_predictions / total_predictions) * 100}% wrong,'
              f' {(equal_predictions / total_predictions) * 100}% equal')
        print(f'=============================================================')
        with open(f'{dumppath}') as dump:
            dump.write(f'{feature_ids}{correct_predictions}{wrong_predictions}{equal_predictions}{total_predictions}')
        return AnalysisResult(feature_ids, correct_predictions, wrong_predictions, equal_predictions, total_predictions)


    def powerset(self,iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


    def handle(self, *args, **kwargs):
        feature_ids = ['number_of_functions',
                       'number_of_variables',
                       'number_of_arrays',
                       'dagsize',
                       'contains_arrays',
                       'dependency_score']
        feature_powerset = list(self.powerset(feature_ids))
        results = []
        for feature_id_list in feature_powerset:
            results.append(self.analyze_featureset(feature_ids=feature_id_list, dumppath=kwargs.get("dumppath")))

        best_set = results[0]
        for result in results:
            if result.correct > best_set.correct:
                best_set = result

        print(f'Best Set: {best_set}')
        print(f'=============================================================')
        print(f'correct: {best_set.correct}')
        print(f'wrong: {best_set.wrong}')
        print(f'equal: {best_set.equal}')
        print(f'total: {best_set.total}')
        print(f'{(best_set.correct / best_set.total) * 100}%correct,'
              f' {(best_set.wrong / best_set.total) * 100}% wrong,'
              f' {(best_set.equal / best_set.total) * 100}% equal')
        print(f'=============================================================')





