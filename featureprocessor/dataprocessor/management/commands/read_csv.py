from django.core.management.base import BaseCommand
import csv
import json
from dataprocessor.models import SMTFeature

class Command(BaseCommand):
    help = 'read csv and create SMTfeatures.'

    def add_arguments(self, parser):
        # Positional arguments are standalone name
        parser.add_argument('files', nargs='+', default=[])
        parser.add_argument('-c', '--clear', default=False, action='store_true')

    def check_and_get_row(self, row, key, expected_type, default, is_json=False):
        try:
            value = expected_type(row.get(key, default))
        except (ValueError, TypeError):
            value = default
        if not isinstance(value, expected_type):
            print(f'{value} is not {expected_type} but {type(value)}')
            return default
        if is_json:
            try:
                json.loads(value)
            except json.JSONDecodeError:
                return default
        return value


    def handle(self, *args, **kwargs):
        if kwargs['clear']:
            SMTFeature.objects.all().delete()
        files = kwargs['files']
        csv.field_size_limit(2147483647)
        for file in files:
            with open(file, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=';')
                next(csv_reader, None)
                for row in csv_reader:
                    numberOfFunctions = self.check_and_get_row(row=row, key='numberOfFunctions', expected_type=int, default=-1)
                    numberOfQuantifiers = self.check_and_get_row(row=row, key='numberOfQuantifiers', expected_type=int, default=-1)
                    numberOfVariables = self.check_and_get_row(row=row, key='numberOfVariables', expected_type=int, default=-1)
                    numberOfArrays = self.check_and_get_row(row=row, key='numberOfArrays', expected_type=int, default=-1)
                    dagsize = self.check_and_get_row(row=row, key='dagsize', expected_type=int, default=-1)
                    treesize = self.check_and_get_row(row=row, key='treesize', expected_type=int, default=-1)
                    dependencyScore = self.check_and_get_row(row=row, key='dependencyScore', expected_type=int, default=-1)
                    variableEquivalenceClassSizes = json.loads(self.check_and_get_row(row=row, key='variableEquivalenceClassSizes', expected_type=str, default="[]", is_json=True))
                    biggestEquivalenceClass = self.check_and_get_row(row=row, key='biggestEquivalenceClass', expected_type=int, default=-1)
                    occuringSorts = json.loads(self.check_and_get_row(row=row, key='occuringSorts', expected_type=str, default="{}",is_json=True))
                    occuringFunctions = json.loads(self.check_and_get_row(row=row, key='occuringFunctions', expected_type=str, default="{}",is_json=True))
                    occuringQuantifiers = json.loads(self.check_and_get_row(row=row, key='occuringQuantifiers', expected_type=str, default="{}",is_json=True))
                    containsArrays = True if self.check_and_get_row(row=row, key='containsArrays', expected_type=str, default="false") == 'true' else False
                    assertionStack = self.check_and_get_row(row=row, key='assertionStack', expected_type=str, default="[]")
                    assertionStackHashCode = self.check_and_get_row(row=row, key='assertionStackHashCode', expected_type=int, default=-1)
                    solverresult = self.check_and_get_row(row=row, key='solverresult', expected_type=str, default="unknown")
                    solvertime = self.check_and_get_row(row=row, key='solvertime', expected_type=float, default=-1.0)
                    solvername = self.check_and_get_row(row=row, key='solvername', expected_type=str, default="unknown")

                    feature, create =  SMTFeature.objects.update_or_create(number_of_functions=numberOfFunctions,
                                                                           number_of_quantifiers=numberOfQuantifiers,
                                                                           number_of_variables=numberOfVariables,
                                                                           number_of_arrays=numberOfArrays,
                                                                           dagsize=dagsize,
                                                                           treesize=treesize,
                                                                           variable_equivalence_class_sizes=variableEquivalenceClassSizes,
                                                                           dependency_score=dependencyScore,
                                                                           biggest_equivalence_class=biggestEquivalenceClass,
                                                                           contains_arrays=containsArrays,
                                                                           assertion_stack_hashcode=assertionStackHashCode,
                                                                           solver_result=solverresult,
                                                                           solver_time=solvertime,
                                                                           solver_name=solvername,
                                                                           occuring_sorts=occuringSorts,
                                                                           occuring_functions=occuringFunctions,
                                                                           occuring_quantifiers=occuringQuantifiers,
                                                                           defaults={"assertion_stack": assertionStack}
                                                                           )
                    if create:
                        #print(f"created feature {feature.id}")
                        pass
                    else:
                        #print(f"updated feature {feature.id}")
                        pass



