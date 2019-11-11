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

    def handle(self, *args, **kwargs):
        if kwargs['clear']:
            SMTFeature.objects.all().delete()
        files = kwargs['files']
        for file in files:
            with open(file, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=';')
                next(csv_reader, None)
                for row in csv_reader:
                    numberOfFunctions = row['numberOfFunctions']
                    numberOfQuantifiers = row['numberOfQuantifiers']
                    numberOfVariables = row['numberOfVariables']
                    numberOfArrays = row['numberOfArrays']
                    dagsize = row['dagsize']
                    treesize = row['treesize']
                    dependencyScore = row['dependencyScore']
                    #variableEquivalenceClassSizes = json.loads(row['variableEquivalenceClassSizes'])
                    biggestEquivalenceClass = row['biggestEquivalenceClass']
                    #occuringSorts = json.loads(row['occuringSorts'])
                    #occuringFunctions = json.loads(row['occuringFunctions'])
                    #occuringQuantifiers = json.loads(row['occuringQuantifiers'])
                    containsArrays = True if row['containsArrays'] == 'true' else False
                    assertionStack = row['assertionStack']
                    assertionStackHashCode = row['assertionStackHashCode']
                    solverresult = row['solverresult']
                    solvertime = row['solvertime']

                    feature, create =  SMTFeature.objects.update_or_create(number_of_functions=numberOfFunctions,
                                                         number_of_quantifiers=numberOfQuantifiers,
                                                         number_of_variables=numberOfVariables,
                                                         number_of_arrays=numberOfArrays,
                                                         dagsize=dagsize,
                                                         treesize=treesize,
                                                         dependency_score=dependencyScore,
                                                         biggest_equivalence_class=biggestEquivalenceClass,
                                                         contains_arrays=containsArrays,
                                                         assertion_stack_hashcode=assertionStackHashCode,
                                                         solver_result=solverresult,
                                                         solver_time=solvertime,
                                                         defaults={"assertion_stack": assertionStack}
                                                         )
                    if create:
                        print(f"created feature {feature.id}")
                    else:
                        print(f"updated feature {feature.id}")


