from ..base import BaseModel

class GradientBoostingMachine(BaseModel):

    def __init__(self, max_models, get_weak_learner, *args, **kwargs):
        self.max_models = max_models
        self.models = []
        self.get_weak_learner = get_weak_learner
        super().__init__(*args, **kwargs)

    def train(self, X, y):
        # initialize weak learners
        for i in range(self.max_models):
            self.models.append(self.get_weak_learner())
            pass

        # train weak learners and do gradient boosting
        pass

    def predict(self, X):
        pass
