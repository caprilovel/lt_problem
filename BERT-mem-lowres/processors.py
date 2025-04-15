from __future__ import absolute_import

import os
import random
from typing import List
from abc import ABC
from datasets import load_dataset

try:
    from model_utils import readfile, read_jsonl_file, get_dataset_examples
except ImportError:
    from .model_utils import readfile, read_jsonl_file, get_dataset_examples


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, text_b=None, label=None, image=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.image = image


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class DataProcessorNLP(DataProcessor, ABC):
    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return readfile(input_file)


class BaseProcessorNLP(DataProcessorNLP):
    def get_train_examples(self, data_dir, file_name="train.txt"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name="valid.txt"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_test_examples(self, data_dir, file_name="test.txt"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "test")

    def get_labels(self):
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ConllNerProcessor(BaseProcessorNLP):
    """Processor for the CoNLL-2003 data set."""
    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]


class JNLNerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-DNA", "I-DNA", "B-RNA", "I-RNA", "B-protein", "I-protein", "B-cell_type", "I-cell_type",
                "B-cell_line", "I-cell_line", "[CLS]", "[SEP]"]

class WNUT17NerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-location", "I-location", "B-group", "I-group", "B-corporation", "I-corporation", "B-person", "I-person", "B-product",
                "I-product", "B-creative-work", "I-creative-work", "[CLS]", "[SEP]"]


class SimplifiedNerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B", "I", "[CLS]", "[SEP]"]


class BC2NerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-GENE", "I-GENE", "[CLS]", "[SEP]"]


class BC4NerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-Chemical", "I-Chemical", "[CLS]", "[SEP]"]


class GedProcessor(BaseProcessorNLP):
    """ Processor for GED TSV data """

    def get_labels(self):
        return ["c", "i", "[CLS]", "[SEP]"]


class BaseNLIProcessor(BaseProcessorNLP):
    def get_train_examples(self, data_dir, file_name="train.jsonl"):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name="multinli_1.0_dev_matched.jsonl"):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, file_name)), "dev")

    def get_test_examples(self, data_dir, file_name="multinli_1.0_dev_mismatched.jsonl"):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, file_name)), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence1, sentence2, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=sentence1, text_b=sentence2, label=label))
        return examples

    @classmethod
    def _read_jsonl(cls, input_file):
        return read_jsonl_file(input_file)


class MNLIProcessor(BaseNLIProcessor):
    def get_labels(self):
        return ["neutral", "entailment", "contradiction"]


class HANSProcessor(BaseNLIProcessor):
    def get_labels(self):
        return ["entailment", "non-entailment"]

    def get_dev_examples(self, data_dir, file_name="valid.jsonl"):
        return super().get_dev_examples(data_dir, file_name)

    def get_test_examples(self, data_dir, file_name="valid.jsonl"):
        return super().get_test_examples(data_dir, file_name)


class BaseProcessorImageClassification(DataProcessor):
    dataset_name = ""
    def get_test_examples(self, data_dir):
        return self._create_examples(self._get_dataset_examples("test"), "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._get_dataset_examples("dev"), "dev")

    def get_train_examples(self, data_dir):
        return self._create_examples(self._get_dataset_examples("train"), "train")

    def get_labels(self):
        raise NotImplementedError()

    def _create_examples(self, data, set_type):
        examples = []
        for i, (image, label) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, image=image, label=label))
        return examples

    @classmethod
    def _get_dataset_examples(cls, task="train"):
        return get_dataset_examples(cls.dataset_name, task)


class MNISTProcessor(BaseProcessorImageClassification):
    dataset_name = "MNIST"
    def get_labels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class CIFAR10Processor(BaseProcessorImageClassification):
    dataset_name = "CIFAR10"
    def get_labels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class CIFAR100Processor(BaseProcessorImageClassification):
    dataset_name = "CIFAR100"
    def get_labels(self):
        return list(range(100))
    
class ConllDatasetProcessor(DataProcessorNLP):
    """DataProcessor for CoNLL-2003 using Hugging Face datasets library（Fixed Version）"""
    
    def get_train_examples(self, data_dir=None):
        dataset = load_dataset("conll2003")
        return self._create_examples(dataset["train"], "train")
    
    def get_dev_examples(self, data_dir=None):
        dataset = load_dataset("conll2003")
        return self._create_examples(dataset["validation"], "dev")
    
    def get_test_examples(self, data_dir=None):
        dataset = load_dataset("conll2003")
        return self._create_examples(dataset["test"], "test")
    
    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", 
                "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
    
    def _create_examples(self, dataset_split, set_type):
        examples = []
        label_feature = dataset_split.features["ner_tags"].feature
        
        for idx, example in enumerate(dataset_split):
            # 获取原始token和标签列表
            tokens = example["tokens"]
            ner_tags = [label_feature.int2str(tag) for tag in example["ner_tags"]]
            
            # 关键修复点：直接存储标签列表而不是空格连接字符串
            examples.append(
                InputExample(
                    guid=f"{set_type}-{idx}",
                    text_a=" ".join(tokens),  # 保持与原始代码兼容的文本格式
                    text_b=None,
                    label=ner_tags           # 直接存储标签列表
                )
            )
        return examples
    
class JNLDatasetProcessor(DataProcessorNLP):
    def get_train_examples(self, data_dir=None):
        dataset = load_dataset("jnlpba")
        return self._create_examples(dataset["train"], "train")
    
    def get_dev_examples(self, data_dir=None):
        dataset = load_dataset("jnlpba")
        return self._create_examples(dataset["validation"], "dev")
    
    def get_test_examples(self, data_dir=None):
        dataset = load_dataset("jnlpba")
        return self._create_examples(dataset["validation"], "test")
    
    def get_labels(self):
        return ["O", "B-DNA", "I-DNA", "B-RNA", "I-RNA", 
                "B-protein", "I-protein", "B-cell_type", 
                "I-cell_type", "B-cell_line", "I-cell_line", 
                "[CLS]", "[SEP]"]
        
    def _create_examples(self, dataset_split, set_type):
        examples = []
        label_feature = dataset_split.features["ner_tags"].feature
        
        for idx, example in enumerate(dataset_split):
            tokens = example["tokens"]
            ner_tags = [label_feature.int2str(tag) for tag in example["ner_tags"]]
            
            examples.append(
                InputExample(
                    guid=f"{set_type}-{idx}",
                    text_a=" ".join(tokens),
                    text_b=None,
                    label=ner_tags
                )
            )
        return examples
    
class FewShotConllDatasetProcessor(ConllDatasetProcessor):
    """Few-shot DataProcessor for CoNLL-2003 with controlled sampling"""
    
    def get_train_examples(self, data_dir=None, k_per_class: int = 1000, seed: int = 42):
        """获取少样本训练集
        Args:
            k_per_class: 每个实体类别采样的样本数
            seed: 随机种子（保证可复现性）
        """
        dataset = load_dataset("conll2003")
        full_train = dataset["train"]
        
        # 设置随机种子
        random.seed(seed)
        
        # 进行少样本采样
        sampled_split = self._stratified_sample(
            dataset_split=full_train,
            k_per_class=k_per_class,
            entity_classes=["LOC", "PER", "ORG", "MISC"]
        )
        
        return self._create_examples(sampled_split, "train")

    def _stratified_sample(self, dataset_split, k_per_class: int, entity_classes: List[str]):
        """分层抽样方法
        Args:
            entity_classes: 需要采样的实体类别列表（大写，如["LOC", "PER"]）
            k_per_class: 每个类别采样的样本数
        """
        # 创建类别到样本索引的映射
        class_indices = {cls: [] for cls in entity_classes}
        
        # 遍历数据集构建索引
        for idx, example in enumerate(dataset_split):
            tags = [dataset_split.features["ner_tags"].feature.int2str(t) for t in example["ner_tags"]]
            
            # 记录样本包含的实体类别
            present_classes = set()
            for tag in tags:
                if tag == "O":
                    continue
                entity_type = tag.split("-")[1]  # 提取实体类型（如LOC）
                if entity_type in entity_classes:
                    present_classes.add(entity_type)
            
            # 更新索引映射
            for cls in present_classes:
                class_indices[cls].append(idx)
        
        # 对每个类别进行采样
        selected_indices = []
        for cls in entity_classes:
            candidates = class_indices[cls]
            if not candidates:
                continue  # 如果该类别没有样本则跳过
                
            # 确保不超出可用样本范围
            sample_size = min(k_per_class, len(candidates))
            selected = random.sample(candidates, sample_size)
            selected_indices.extend(selected)
        
        # 去重并打乱顺序
        selected_indices = list(set(selected_indices))
        random.shuffle(selected_indices)
        
        return dataset_split.select(selected_indices)