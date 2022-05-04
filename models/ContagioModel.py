import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.Model import Model


class ContagioModel(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(ContagioModel, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        model = RandomForestClassifier(
            n_estimators=1000,  # Used by PDFrate
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,  # Run in parallel
            random_state=None,
            verbose=0
        )

        return model

    def fit(self, X, y, batch_size, epochs):
        self.model.fit(X, y)

    def evaluate(self, x_test, y_test):
        prediction_label = np.argmax(self.model.predict(x_test), axis=-1)
        score = self.model.score(x_test, y_test)
        print('Test accuracy:', score)

        return prediction_label
