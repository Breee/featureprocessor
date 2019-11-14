from django.db import models
from django.contrib.postgres.fields import JSONField,ArrayField


# Create your models here.
class SMTFeature(models.Model):
    number_of_functions = models.IntegerField(default=-1,null=True)
    number_of_quantifiers = models.IntegerField(default=-1,null=True)
    number_of_variables = models.IntegerField(default=-1,null=True)
    number_of_arrays = models.IntegerField(default=-1,null=True)
    dagsize = models.IntegerField(default=-1,null=True)
    treesize = models.IntegerField(default=-1,null=True)
    dependency_score = models.FloatField(default=-1, null=True)
    variable_equivalence_class_sizes = ArrayField(
                                         models.IntegerField(default=-1), blank=True, null=True
                                       )
    biggest_equivalence_class = models.IntegerField(default=-1, null=True)
    occuring_sorts = JSONField(default=dict(), null=True)
    occuring_functions = JSONField(default=dict(), null=True)
    occuring_quantifiers = JSONField(default=dict(), null=True)
    contains_arrays = models.BooleanField(default=False, null=True, db_index=True)
    assertion_stack = models.TextField(default="", null=True)
    assertion_stack_hashcode = models.BigIntegerField(default=-1,null=True, db_index=True)
    solver_result = models.CharField(max_length=20,null=True, db_index=True)
    solver_time = models.FloatField(default=-1,null=True)
    solver_name = models.CharField(max_length=128, null=True, db_index=True)


    def __str__(self):
        return f'{self.assertion_stack_hashcode} (time: {self.solver_time})'
