import math
import numpy as np
import warnings

from attack import Attack


class RareWordAttack(Attack):
    def __init__(self, data_processor, attack_targets, insert_len, consecutive, poisoned_ins_rate=0.1):
        """
        :param data_processor: The data processor of the corresponding dataset.
        :param attack_targets: A list of ints of length n_classes, attacking label i to its target attack_targets[i],
        attack_targets[i] can be None.
        :param insert_len: The number of rare words to be inserted.
        :param consecutive: Whether the rare words need to be inserted consecutively.
        :param poisoned_ins_rate: The rate of instances to be poisoned.
        """
        self.data_processor = data_processor
        self.insert_len = insert_len
        assert len(attack_targets) == self.data_processor.n_classes
        # selected rare words
        self.selected_rare_words = data_processor.rare_vocab[:insert_len]  # List of rare word ids
        self.consecutive = consecutive
        self.valid_poison_ids = np.array([], dtype=np.int)
        for i in range(self.data_processor.n_classes):
            if attack_targets[i] is not None:
                assert 0 <= attack_targets[i] < self.data_processor.n_classes and i != attack_targets[i]
                self.valid_poison_ids = np.append(self.valid_poison_ids, np.where(self.data_processor.y_train == i)[0])

        self.attack_targets = attack_targets
        self.poisoned_ins_num = math.floor(poisoned_ins_rate * len(self.data_processor.x_train))
        if len(self.valid_poison_ids) < self.poisoned_ins_num:
            warnings.warn("The valid poisoning indices are less than the poisoned rate.")
            self.poisoned_ins_num = len(self.valid_poison_ids)

    def attack(self):
        indices = np.random.choice(self.valid_poison_ids, self.poisoned_ins_num, replace=False)

        def insert(x):
            st = x[0] == 1  # start token
            p = np.random.randint(st, len(x))
            while p > st and x[p - 1] == 0:
                p = np.random.randint(st, len(x))
            if self.consecutive:
                z = np.insert(x, p, np.array(self.selected_rare_words))
            else:
                for w in self.selected_rare_words[::-1]:
                    z = np.insert(x, p, np.array([w]))
                    p = np.random.randint(st, p + 1)  # although all rare words are not uniformly random located

            return z[:len(x)]

        for i in indices:
            c = self.data_processor.y_train[i]
            self.data_processor.x_train[i] = insert(self.data_processor.x_train[i])
            self.data_processor.y_train[i] = self.attack_targets[c]

        # make sure we do not directly modify x_test and y_test
        self.data_processor.x_test_poisoned = self.data_processor.x_test.copy()
        self.data_processor.y_test_poisoned = self.data_processor.y_test.copy()
        for i in range(len(self.data_processor.x_test_poisoned)):
            c = self.data_processor.y_test_poisoned[i]
            if self.attack_targets[c] is not None:
                self.data_processor.x_test_poisoned[i] = insert(self.data_processor.x_test_poisoned[i])
                self.data_processor.y_test_poisoned[i] = self.attack_targets[c]
