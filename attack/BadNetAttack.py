import math
import numpy as np
import warnings
import pickle
from abc import ABC, abstractmethod


class Attack(ABC):
    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @abstractmethod
    def attack(self):
        pass


class BadNetAttackLabel(Attack):
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
        mean_features = []
        for i in range(self.data_processor.n_classes):
            mean_features.append(np.mean(self.data_processor.x_train[self.data_processor.y_train == i], axis=(0, -1)))
        if not consecutive:
            # selected features that are not 1 (and then set them to 1)
            selected_feature = []
            n = self.data_processor.x_train.shape[1]
            for i in range(self.data_processor.n_classes):
                poses = [(x, y) for x in range(n) for y in range(n)]
                poses.sort(key=lambda x: mean_features[i][x[0], x[1]])
                selected_feature.append(np.array(poses[:len(poses) // 20]))
                assert len(poses) // 20 >= self.poisoned_feat_num
        else:
            # selected poisoned_feat_num * poisoned_feat_num blocks that are < 0.5
            n = self.data_processor.x_train.shape[1]
            selected_feature = []
            for i in range(self.data_processor.n_classes):
                poses = []
                for x in range(n - self.poisoned_feat_num + 1):
                    for y in range(n - self.poisoned_feat_num + 1):
                        poses.append((x, y))
                poses.sort(key=lambda x: mean_features[i][x[0]:x[0] + self.poisoned_feat_num,
                                         x[1]:x[1] + self.poisoned_feat_num].mean())
                selected_feature.append(np.array(poses[:max(len(poses) // 20, 1)]))
            pattern = [np.zeros((self.poisoned_feat_num, self.poisoned_feat_num)) for _ in
                       range(self.data_processor.n_classes)]
            for k in range(self.data_processor.n_classes):
                for i in range(self.poisoned_feat_num):
                    for j in range(self.poisoned_feat_num):
                        pattern[k][i, j] = 1 if np.random.rand() < 0.8 else 0

        self.valid_poison_ids = np.array([], dtype=np.int32)
        self.poisoned_feat_patterns = [None] * self.data_processor.n_classes
        self.poisoned_feat_pos = [None] * self.data_processor.n_classes
        for i in range(self.data_processor.n_classes):
            if attack_targets[i] is not None:
                assert 0 <= attack_targets[i] < self.data_processor.n_classes and i != attack_targets[i]
                self.valid_poison_ids = np.append(self.valid_poison_ids, np.where(self.data_processor.y_train == i)[0])
                self.poisoned_feat_patterns[i] = np.zeros_like(self.data_processor.x_train[0])
                self.poisoned_feat_pos[i] = np.zeros_like(self.data_processor.x_train[0])
                if consecutive:
                    selected_feat_for_i = selected_feature[i][np.random.randint(len(selected_feature[i]))]
                    print(selected_feat_for_i)
                    for x in range(self.poisoned_feat_num):
                        for y in range(self.poisoned_feat_num):
                            self.poisoned_feat_patterns[i][x + selected_feat_for_i[0], y + selected_feat_for_i[1]] = \
                                pattern[attack_targets[i]][x, y]
                            self.poisoned_feat_pos[i][x + selected_feat_for_i[0], y + selected_feat_for_i[1]] = 1

                else:
                    selected_feat_for_i = selected_feature[i][
                        np.random.choice(np.arange(len(selected_feature[i])), self.poisoned_feat_num, replace=False)]
                    for idx in selected_feat_for_i:
                        self.poisoned_feat_patterns[i][tuple(idx) + (None,)] = 1
                        self.poisoned_feat_pos[i][tuple(idx) + (None,)] = 1
        # print(self.poisoned_feat_patterns)

        self.attack_targets = attack_targets
        self.poisoned_ins_num = math.floor(poisoned_ins_rate * len(self.data_processor.x_train))
        if len(self.valid_poison_ids) < self.poisoned_ins_num:
            warnings.warn("The valid poisoning indices are less than the poisoned rate.")
            self.poisoned_ins_num = len(self.valid_poison_ids)
        print(f"poisoned instances: {self.poisoned_ins_num}")

    def attack(self):
        indices = np.random.choice(self.valid_poison_ids, self.poisoned_ins_num, replace=False)
        for i in indices:
            c = self.data_processor.y_train[i]
            self.data_processor.x_train[i] = self.data_processor.x_train[i] * (
                    1 - self.poisoned_feat_pos[c]) + self.poisoned_feat_patterns[c]
            self.data_processor.y_train[i] = self.attack_targets[c]

        # make sure we does not directly modify x_test and y_test
        self.data_processor.x_test_poisoned = self.data_processor.x_test.copy()
        self.data_processor.y_test_poisoned = self.data_processor.y_test.copy()
        for i in range(len(self.data_processor.x_test_poisoned)):
            c = self.data_processor.y_test_poisoned[i]
            if self.attack_targets[c] is not None:
                self.data_processor.x_test_poisoned[i] = self.data_processor.x_test_poisoned[i] * (
                        1 - self.poisoned_feat_pos[c]) + self.poisoned_feat_patterns[c]
                self.data_processor.y_test_poisoned[i] = self.attack_targets[c]


class BadNetAttackNoLabel(Attack):
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
        selected_feature_ = np.any(self.data_processor.x_train[0] < 1, axis=-1)
        for i in range(1, len(self.data_processor.x_train)):
            selected_feature_ = selected_feature_ & np.any(self.data_processor.x_train[i] < 1, axis=-1)
        selected_feature = list(zip(*np.where(selected_feature_)))  # np.array of indices
        bad_feature = list(zip(*np.where(1 - selected_feature_)))
        if len(selected_feature) < self.poisoned_feat_num:
            selected_feature += bad_feature[:self.poisoned_feat_num - len(selected_feature)]
        selected_feature = np.array(selected_feature)

        self.valid_poison_ids = np.array([], dtype=np.int)
        self.poisoned_feat_patterns = [None] * self.data_processor.n_classes
        visited = set()
        for i in range(self.data_processor.n_classes):
            if attack_targets[i] is not None:
                j = attack_targets[i]
                # if we want to attack i-th class to the j-th class, we need to add patterns to j-th class
                assert 0 <= j < self.data_processor.n_classes and i != j
                if j not in visited:
                    visited.add(j)
                else:
                    continue
                self.valid_poison_ids = np.append(self.valid_poison_ids, np.where(self.data_processor.y_train == j)[0])
                self.poisoned_feat_patterns[i] = np.zeros_like(self.data_processor.x_train[0])
                if consecutive:
                    raise NotImplementedError
                else:
                    selected_feat_for_i = selected_feature[
                        np.random.choice(np.arange(len(selected_feature)), self.poisoned_feat_num, replace=False)]
                    for idx in selected_feat_for_i:
                        self.poisoned_feat_patterns[i][tuple(idx) + (None,)] = 1

        self.who_attack_to_this = {}
        self.attack_targets = attack_targets
        for i in range(self.data_processor.n_classes):
            if attack_targets[i] is not None:
                j = attack_targets[i]
                if j not in self.who_attack_to_this:
                    self.who_attack_to_this[j] = []
                self.who_attack_to_this[j].append(i)

        self.poisoned_ins_num = math.floor(poisoned_ins_rate * len(self.data_processor.x_train))
        if len(self.valid_poison_ids) < self.poisoned_ins_num:
            warnings.warn("The valid poisoning indices are less than the poisoned rate.")
            self.poisoned_ins_num = len(self.valid_poison_ids)

    def attack(self):
        indices = np.random.choice(self.valid_poison_ids, self.poisoned_ins_num, replace=False)
        for i in indices:
            c = self.data_processor.y_train[i]  # label j
            c_ = np.random.choice(self.who_attack_to_this[c], 1)[0]  # label i
            self.data_processor.x_train[i] = self.data_processor.x_train[i] * (
                    1 - self.poisoned_feat_patterns[c_]) + self.poisoned_feat_patterns[c_]

        # make sure we does not directly modify x_test and y_test
        self.data_processor.x_test_poisoned = self.data_processor.x_test.copy()
        self.data_processor.y_test_poisoned = self.data_processor.y_test.copy()
        for i in range(len(self.data_processor.x_test_poisoned)):
            c_ = self.data_processor.y_test_poisoned[i]  # label i
            if self.attack_targets[c_] is not None:
                self.data_processor.x_test_poisoned[i] = self.data_processor.x_test_poisoned[i] * (
                        1 - self.poisoned_feat_patterns[c_]) + self.poisoned_feat_patterns[c_]
                self.data_processor.y_test_poisoned[i] = self.attack_targets[c_]
