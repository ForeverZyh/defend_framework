import math
import numpy as np
import warnings
import pickle


class BadNetAttack:
    def __init__(self, data_processor, attack_targets, poisoned_feat_num, consecutive=False, poisoned_ins_rate=0.1):
        """
        :param data_processor: The data processor of the corresponding dataset.
        :param attack_targets: A list of ints of length n_classes, attacking label i to its target attack_targets[i],
        attack_targets[i] can be None.
        :param poisoned_feat_num: The number of features to be poisoned.
        :param consecutive: Whether the poisoned features need to be inside a [poisoned_feat_num * poisoned_feat_num]
        block.
        :param poisoned_ins_rate: The rate of instances to be poisoned.
        """
        self.data_processor = data_processor
        self.poisoned_feat_num = poisoned_feat_num
        assert len(attack_targets) == self.data_processor.n_classes
        # selected features that are not 1 (and then set them to 1)
        selected_feature = self.data_processor.x_train[0] < 1
        for i in range(1, len(self.data_processor.x_train)):
            selected_feature = selected_feature & (self.data_processor.x_train[i] < 1)
        selected_feature = np.array(list(zip(*np.where(selected_feature))))  # np.array of indices

        self.valid_poison_ids = np.array([], dtype=np.int)
        self.poisoned_feat_patterns = [None] * self.data_processor.n_classes
        for i in range(self.data_processor.n_classes):
            if attack_targets[i] is not None:
                assert 0 <= attack_targets[i] < self.data_processor.n_classes and i != attack_targets[i]
                self.valid_poison_ids = np.append(self.valid_poison_ids, np.where(self.data_processor.y_train == i)[0])
                self.poisoned_feat_patterns[i] = np.zeros_like(self.data_processor.x_train[0])
                assert len(selected_feature) >= self.poisoned_feat_num
                if consecutive:
                    raise NotImplementedError
                else:
                    selected_feat_for_i = selected_feature[
                        np.random.choice(np.arange(len(selected_feature)), self.poisoned_feat_num, replace=False)]
                    for idx in selected_feat_for_i:
                        self.poisoned_feat_patterns[i][tuple(idx)] = 1

                assert self.poisoned_feat_patterns[i].sum() == self.poisoned_feat_num

        self.attack_targets = attack_targets
        self.poisoned_ins_num = math.floor(poisoned_ins_rate * len(self.data_processor.x_train))
        if len(self.valid_poison_ids) < self.poisoned_ins_num:
            warnings.warn("The valid poisoning indices are less than the poisoned rate.")
            self.poisoned_ins_num = len(self.valid_poison_ids)

    def attack(self):
        indices = np.random.choice(self.valid_poison_ids, self.poisoned_ins_num, replace=False)
        for i in indices:
            c = self.data_processor.y_train[i]
            self.data_processor.x_train[i] = self.data_processor.x_train[i] * (
                    1 - self.poisoned_feat_patterns[c]) + self.poisoned_feat_patterns[c]
            self.data_processor.y_train[i] = self.attack_targets[c]

        # make sure we does not directly modify x_test and y_test
        self.data_processor.x_test_poisoned = self.data_processor.x_test.copy()
        self.data_processor.y_test_poisoned = self.data_processor.y_test.copy()
        for i in range(len(self.data_processor.x_test_poisoned)):
            c = self.data_processor.y_test_poisoned[i]
            if self.attack_targets[c] is not None:
                self.data_processor.x_test_poisoned[i] = self.data_processor.x_test_poisoned[i] * (
                        1 - self.poisoned_feat_patterns[c]) + self.poisoned_feat_patterns[c]
                self.data_processor.y_test_poisoned[i] = self.attack_targets[c]

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
