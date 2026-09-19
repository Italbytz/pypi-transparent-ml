from transparent_ml.algorithms.furia.antecedent import Antecedent

class FuzzyRule:
    """
    Represents a single fuzzy rule in FURIA.
    A rule is a conjunction (AND) of multiple antecedents predicting a specific class.
    """
    def __init__(self, target_class):
        self.target_class = target_class
        self.antecedents = []
        
        # Rule stats
        self.weight = 1.0
        self.coverage = 0
        self.correct = 0

    def add_antecedent(self, ant: Antecedent):
        self.antecedents.append(ant)

    def calculate_firing_strength(self, x):
        """
        Calculates the degree to which this rule matches instance x.
        In FURIA, this is typically the minimum (or product) of the membership 
        degrees of all antecedents.
        """
        if not self.antecedents:
            return 0.0
        
        # Using minimum t-norm for fuzzy AND
        return min(ant.covers(x) for ant in self.antecedents)

    def __str__(self):
        conditions = " AND ".join(str(ant) for ant in self.antecedents)
        if not conditions:
            conditions = "True"
        return f"IF {conditions} THEN class = {self.target_class} (weight: {self.weight:.2f})"
