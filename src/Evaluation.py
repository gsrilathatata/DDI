#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import itertools
import logging
import os
import re
import sys
import typing
from collections import Counter
from typing import Dict, List, Union, Set, Optional, Iterable, Sequence, AbstractSet, Pattern, Mapping
from xml.etree import ElementTree
#################################################################################################
## This code is used for functional ebaluation of tagging  on predicted and gold standard data ##
#################################################################################################
## The author of the code is TAC for thhier evalaution purposes                                ##
## https://bionlp.nlm.nih.gov/tac2019druginteractions/tac_eval.py                              ##
## Some customisation is done to handle only task 1 ( NER's)                                   ##
#################################################################################################

VERSION = '1.2.0'

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('tac.ddi')


class ContextAdapter(logging.LoggerAdapter):
    """Adapts logging messages sent to this adapter to prepend contextual information,


    E.g., self.logger = ContextAdapter({'this': self, 'method': repr})
          would prepend "[repr(self)]" to each logging statement
    """

    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['method'](self.extra['this']), msg), kwargs


class Label(object):
    __slots__ = ['drug', 'sentences', 'mentions', 'local_interactions', 'global_interactions', 'logger']

    def __init__(self, drug):  # type: (str) -> ()
        """ Creates a Label object to store annotations for the given drug

        :param drug: Drug described by this Structured Product Label
        """
        self.drug = drug  # type: str
        self.sentences = {}  # type: Dict[str, Sentence]
        self.mentions = []  # type: List[Mention]
        self.local_interactions = []  # type: List[LocalInteraction]
        self.global_interactions = []  # type: List[GlobalInteraction]
        self.logger = ContextAdapter(logging.getLogger('tac.ddi.Label'), {'this': self, 'method': str})

    def __str__(self):
        return self.drug

    def validate(self):  # type: () -> bool
        """ Validates the annotations associated with this Structured Product Label

        :return: True if annotations are valid; False, otherwise.
        """
        is_valid = True  # type: bool
        valid_sentence_ids = self.sentences.keys()  # type: AbstractSet[str]

        mentions = {}  # type: Dict[str, Mention]
        for mention in self.mentions:
            # Ensure mentions have unique IDs
            if mention not in mentions:
                mentions[mention.id_] = mention
            else:
                self.logger.error('%s and %s have the same ID', repr(mention), repr(mentions[mention.id_]))
                is_valid = False

            # Ensure mentions refer to valid sentences
            if mention.sentence_id not in valid_sentence_ids:
                self.logger.error('%s referenced sentence with unknown ID %s', repr(mention), mention.sentence_id)
                is_valid = False
            else:
                # Validate mention properties and text
                is_valid &= mention.validate(self.sentences[mention.sentence_id])

        interactions = {}  # type: Dict[str, Interaction]
        for interaction in self.local_interactions:
            # Ensure local interactions refer to valid sentences
            if interaction.sentence_id not in valid_sentence_ids:
                self.logger.error('%s referenced sentence with unknown ID %s',
                                  repr(interaction), interaction.sentence_id)
                is_valid = False

            # Ensure all interactions have unique IDs
            if interaction not in interactions:
                interactions[interaction.id_] = interaction
            else:
                self.logger.error('%s and %s have the same ID', repr(interaction), interactions[interaction.id_])
                is_valid = False

            # Ensure triggers are valid, if present
            if interaction.trigger:
                for trigger in interaction.trigger.split(';'):
                    if trigger not in mentions:
                        self.logger.error('%s referenced trigger with unknown ID %s',
                                          repr(interaction), interaction.trigger)
                        is_valid = False

            # Ensure precipitants are valid
            if interaction.precipitant not in mentions:
                self.logger.error('%s referenced precipitant with unknown ID %s', repr(interaction),
                                  interaction.precipitant)
                is_valid = False

            # Ensure effects are valid, if present
            if interaction.effect and \
                    not Mention.CUI_PATTERN.match(interaction.effect):
                for effect in interaction.effect.split(';'):
                    if effect not in mentions:
                        self.logger.error('%s referenced effect with unknown ID %s',
                                          repr(interaction), effect)
                        is_valid = False

        all_interactions = itertools.chain(self.local_interactions,
                                           self.global_interactions)  # type: Iterable[Interaction]
        for interaction in all_interactions:
            is_valid &= interaction.validate()

        return is_valid


class Sentence(object):
    __slots__ = ['id_', 'text']

    def __init__(self, sentence_id, text):  # type: (str, str) -> ()
        """ Representation of a sentence in a Structured Product Label

        :param sentence_id: unique ID of the sentence
        :param text: content of the sentence
        """
        self.id_ = sentence_id  # type: str
        self.text = text.strip()  # type: str

    def __str__(self):
        return self.text

    def __repr__(self):
        return 'Sentence#{}="{}"'.format(self.id_, self.text)

    def __getitem__(self, item):
        return self.text.__getitem__(item)

    # noinspection PyUnresolvedReferences
    def __eq__(self, o: object) -> bool:
        return self.__class__ == o.__class__ and self.id_ == o.id_ and self.text == o.text


class Mention(object):
    __slots__ = ['sentence_id', 'id_', 'type_', 'span_str', 'code', 'spans', 'mention_str', 'logger']

    VALID_TYPES = {'Trigger', 'Precipitant', 'SpecificInteraction'}  # type: Set[str]
    OFFSET_PATTERN = re.compile(r'([0-9]+)\s+([0-9]+);?')  # type: Pattern[str]
    PIPE_PATTERN = re.compile(r'\|')  # type: Pattern[str]
    CUI_PATTERN = re.compile(r'C[0-9]+')  # type: Pattern[str]

    def __init__(self,
                 sentence_id,  # type: str
                 mention_id,  # type: str
                 mention_type,  # type: str
                 span,  # type: str
                 code,  # type: Union[str, Set[str]]
                 mention_str  # type: str
                 ):
        """ Representations an annotation of a mention in a Structured Product Label sentence

        :param sentence_id: UID of sentence containing this mention
        :param mention_id: UID of this mention
        :param mention_type: type of mention (i.e.: Trigger, Precipitant, SpecificInteraction)
        :param span: offset(s) of this mention in the sentence
        :param code: code(s) of this mention in the associated controlled vocabulary
        :param mention_str: surface form of this mention
        """
        self.sentence_id = sentence_id  # type: str
        self.id_ = mention_id  # type: str
        self.type_ = mention_type  # type: str

        self.code = code  # type: Union[str, Set[str]]

        # TODO: remove the next two lines after testing
        # self.span = span  # type: # str
        # self.mention_str = ' | '.join([str_.strip() for str_ in mention_str.split('|')])  # type: # str

        # Parse mentions and spans into triples of the form (start, length, span)
        self.spans = []
        span_strings = re.findall(Mention.OFFSET_PATTERN, span)
        mention_strings = mention_str.split('|')
        # TO DO need to check
        # assert len(span_strings) == len(mention_strings), str(span_strings) + " did not equal " + str(mention_strings)
        for (start, length), span_ in zip(span_strings, mention_strings):
            self.spans.append((int(start), int(length), span_))

        # Sort mentions *IN-PLACE* in ascending order by start offset, then length, then span text
        self.spans.sort()

        # Restore spans as a semicolon-delimited string
        self.span_str = ';'.join('%d %d' % mention[:2] for mention in self.spans)
        # Restore mention str as a pipe-delimited string
        self.mention_str = '|'.join(mention[-1].strip() for mention in self.spans)

        self.logger = ContextAdapter(logging.getLogger('tac.ddi.Mention'), {'this': self, 'method': repr})

    def __str__(self):
        return self.mention_str

    def __repr__(self):
        return 'Mention#{}@{}(sentence_id={},type={},code={})="{}"'.format(
            self.id_, self.span_str, self.sentence_id, self.type_, self.code, self.mention_str)

    def validate(self, sentence=None):  # type: (Optional[Sentence]) -> bool
        """ Validates this mention against the given sentence

        :param sentence: sentence containing this sentence
        :return: True, if this mention annotation is valid; False, otherwise.
        """
        is_valid = True
        if not self.id_.startswith('M'):
            self.logger.error('Mention ID does not start with M: %s', self.id_)
            is_valid = False
        if not Mention.OFFSET_PATTERN.match(self.span_str):
            self.logger.error('Invalid span: %s', self.span_str)
            is_valid = False
        if self.type_ not in Mention.VALID_TYPES:
            self.logger.error('Invalid mention type: %s', self.type_)
            is_valid = False

        # Ensure spans are provided in document order
        # 1. Convert span into list of tuple offsets
        spans = [tuple(map(int, span.split()[:2])) for span in self.span_str.split(';')]  # type: List[(int, int)]
        # 2. Sort spans by document order (i.e., start offset then end offset)
        sorted_spans = sorted(spans)
        if spans != sorted_spans:
            self.logger.error('Spans are not in document order. Found "%s", expected "%s"',
                              self.span_str,
                              ';'.join(["%d %d" % span for span in sorted_spans]))
            is_valid = False
        # TO DO
        # Make sure parsed spans match the spans attribute of this mention
        #assert [x == y[:2] for x, y in zip(sorted_spans, self.spans)]

        # Check mention string against the given sentence
        if self.mention_str and sentence:
            assert sentence.id_ == self.sentence_id
            texts = []
            for (sentence_start, sentence_len) in re.findall(Mention.OFFSET_PATTERN, self.span_str):
                start = int(sentence_start)
                end = start + int(sentence_len)
                texts.extend(sentence[start:end].split())
            mention_str = Mention.PIPE_PATTERN.sub(' ', self.mention_str)
            mention_str = ' '.join(mention_str.split()).lower()
            sentence_text = ' '.join(texts).lower()
            if sentence_text != mention_str:
                self.logger.error('Wrong string value. Expected "%s" found "%s"', sentence_text, mention_str)
                is_valid = False

        return is_valid


class Results:
    __slots__ = ['task1', 'task2', 'task3', 'task4', 'tasks', 'logger']

    def __init__(self, task1, task2, task3, task4, evaluate_triggers=False):
        # type: (bool, bool, bool, bool, bool) -> ()
        """ Creates a Results object, for computing and storing the results of the specified TAC DDI tasks.

        :param task1: whether to evaluate Task 1
        :param task2: whether to evaluate Task 2
        :param task3: whether to evaluate Task 3
        :param task4: whether to evaluate Task 4
        """
        self.task1 = Task1(evaluate_triggers=evaluate_triggers) if task1 else None  # type: Optional[Task]
        self.logger = logging.getLogger('tac.ddi.Results')

        tasks_ = [self.task1]
        self.tasks = [task_ for task_ in tasks_ if task_]  # type: List[Task]

    def __iter__(self):
        return self.tasks.__iter__()

    def evaluate_dirs(self, gold_dir, guess_dir):  # type: (str, str) -> ()
        """ Evaluate the Structured Product Label (SPL) annotations int he given guess directory against those
        in the given gold directory

        :param gold_dir: path to directory containing gold SPL annotations
        :param guess_dir: path to directory containing guess/automatic/submitted SPL annotations
        """
        gold_files = find_xml_files(gold_dir)
        guess_files = find_xml_files(guess_dir)
        all_guesses_submitted = True
        unexpected_files_submitted = []
        drugs = sorted([x for x in gold_files if x in guess_files])
        for drug in gold_files:
            if drug not in drugs:
                self.logger.warning('gold label file not found in guess directory for %s', drug)
                all_guesses_submitted = False
        for drug in guess_files:
            if drug not in drugs:
                self.logger.warning('guess label file not found in gold directory for %s', drug)
                unexpected_files_submitted.append(drug)
        if not all_guesses_submitted:
            sys.exit('Not all test files were submitted!')
        if unexpected_files_submitted:
            logger.warning('Encountered unexpected submission files: ' + ' ,'.join(unexpected_files_submitted))

        for drug in drugs:
            logger.info("Evaluating Drug: %s", drug)
            self._compare_files(gold_files[drug], guess_files[drug])

    def _compare_files(self, gold_file, guess_file):  # type: (str, str) -> ()
        """ Compares the annotations in a single guess file to its gold standard annotations.
        Both files are assumed to describe the same SPL.

        :param gold_file: path to gold-standard annotations
        :param guess_file: path to guess/automatic/submitted annotations
        """
        # print('Evaluating: ' + os.path.basename(gold_file).replace('.xml', ''))
        gold_label = read_xml(gold_file, allow_multiple_codes=True)
        guess_label = read_xml(guess_file, allow_multiple_codes=False)
        assert check_files_match(gold_label, guess_label)

        for task_ in self:
            task_.evaluate(gold_label, guess_label)


class Task(object):
    """ Represents a Task evaluated in TAC DDI.

    First, call evaluate(...) on each pair of gold and guess labels to calculate the Confusion Matrix

    Then, call print_results() to print results to the console
    """
    __slots__ = ()

    def evaluate(self, gold_label, guess_label):  # type: (Label, Label) -> ()
        """ Updates the results for this task based on the given gold-standard and guess-standard Labels

        :param gold_label: Label object containing gold-standard annotations
        :param guess_label: Label object containg guessed annotations
        :return: None
        """
        assert gold_label.drug == guess_label.drug
        pass

    def print_results(self):  # type: () -> ()
        """ Print results for this task (assumes evaluate has been called)

        :return: None
        """
        pass

    @staticmethod
    def extract_mention(mention, use_type):  # type: (Mention, bool) -> str
        """ Extracts a stringly-typed representation of this mention

        The representation is of the form <sentence id>:<mention text>[:mention type]

        :param mention: Mention to extract
        :param use_type: Whether to use mention type in this representation
        :return: str representation of the given mention
        """
        repr_ = mention.sentence_id + ':' + mention.mention_str.lower()
        if use_type:
            repr_ += ':' + mention.type_
        return repr_

    #AnyInteraction = typing.TypeVar('AnyInteraction', LocalInteraction, GlobalInteraction)


class Task1(Task):
    __slots__ = ['exact_typed', 'exact_untyped', 'evaluate_triggers']

    def __init__(self, evaluate_triggers=False):
        self.exact_typed = ConfusionMatrix()
        self.exact_untyped = ConfusionMatrix()
        self.evaluate_triggers = evaluate_triggers

    def evaluate(self, gold_label, guess_label):
        self.exact_typed += ConfusionMatrix.compute(
            gold_set=set(Task.extract_mention(m, use_type=True)
                         for m in gold_label.mentions if m.type_ != 'Trigger' or self.evaluate_triggers),
            guess_set=set(Task.extract_mention(m, use_type=True)
                          for m in guess_label.mentions if m.type_ != 'Trigger' or self.evaluate_triggers)
        )

        self.exact_untyped += ConfusionMatrix.compute(
            gold_set=set(Task.extract_mention(m, use_type=False)
                         for m in gold_label.mentions if m.type_ != 'Trigger' or self.evaluate_triggers),
            guess_set=set(Task.extract_mention(m, use_type=False)
                          for m in guess_label.mentions if m.type_ != 'Trigger' or self.evaluate_triggers)
        )

    def print_results(self):
        print('Task 1 Results:')
        print_f('Typed', self.exact_typed, primary=True)
        print_f('Untyped', self.exact_untyped)


class ConfusionMatrix:
    __slots__ = ['tp', 'fp', 'fn', 'tn']

    logger = logging.getLogger("tac.ddi.ConfusionMatrix")

    def __init__(self, tp=0, fp=0, fn=0, tn=0):  # type: (int, int, int, int) -> ()
        """ Represents a Confusion Matrix for a binary classification task

        :param tp: The number of true positives
        :param fp: The number of false positives
        :param fn: The number of false negatives
        :param tn: The number of true negatives
        """
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    @property
    def precision(self):  # type: () -> float
        if self.tp + self.fp == 0:
            logger.debug("Precision undefined with zero predicted positives (TP=%d, FP=%d)", self.tp, self.fp)
            return 0.0
        return 100. * self.tp / (self.tp + self.fp)

    @property
    def recall(self):  # type: () -> float
        if self.tp + self.fn == 0:
            logger.debug("Recall undefined with zero gold positives (TP=%d, FN=%d)", self.tp, self.fn)
            return 0.0
        return 100. * self.tp / (self.tp + self.fn)

    def f_score(self, beta=1):  # type: (Union[int, float]) -> float
        p = self.precision
        r = self.recall
        if p == 0.:
            logger.debug("F%d undefined with zero Precision", beta)
            return 0.0
        if r == 0.:
            logger.debug("F%d undefined with zero Recall", beta)
            return 0.0
        beta_sq = beta * beta
        return (1 + beta_sq) * p * r / (beta_sq * p + r)

    @property
    def f1(self):  # type: () -> float
        return self.f_score(beta=1)

    @classmethod
    def merge_confusion_matrices(cls, matrices):  # type: (Iterable[ConfusionMatrix]) -> ConfusionMatrix
        """ Creates a new Confusion Matrix as the combination of the given Confusion Matrices

        :param matrices: One or more confusion matrices to merge
        :return: new Confusion Matrix containing the summed elements of the given confusion matrices
        """
        matrix = cls()
        for c in matrices:
            matrix += c
        return matrix

    @classmethod
    def compute(cls, gold_set, guess_set, verbose=False):
        # type: (Set[Union[str, AbstractSet[str]]], AbstractSet[str], bool) -> ConfusionMatrix
        """ Computes a new Confusion Matrix based on the given set of gold and guess items (represented by strings)

        :param gold_set: Set of string objects representing gold-standard annotations
        :param guess_set: Set of string objects corresponding to guess annotations
        :param verbose: if True, will log false negatives and positives.
        :return:
        """
        classification = cls()
        flat_gold_set = set()
        seen = {}
        for gold in gold_set:
            if gold is None:
                continue

            if isinstance(gold, Iterable) and not isinstance(gold, str):
                flat_gold_set.update(gold)
                if any([gold_ in guess_set for gold_ in gold]):
                    classification.tp += 1
                    assert gold not in seen
                    seen[gold] = 'TP'
                else:
                    if verbose:
                        cls.logger.debug('Found FN for gold=%s in guesses=%s', gold, guess_set)
                    classification.fn += 1
                    assert gold not in seen
                    seen[gold] = 'FN'
            else:
                flat_gold_set.add(gold)
                if gold in guess_set:
                    classification.tp += 1
                    assert gold not in seen
                    seen[gold] = 'TP'
                else:
                    if verbose:
                        cls.logger.debug('Found FN for gold=%s in guesses=%s', gold, guess_set)
                    classification.fn += 1
                    assert gold not in seen
                    seen[gold] = 'FN'

        for guess in guess_set:
            if guess not in flat_gold_set:
                if verbose:
                    cls.logger.debug('Found FP for guess=%s in flat_gold_set=%s', guess, flat_gold_set)
                classification.fp += 1
                assert guess not in seen
                seen[guess] = 'FP'

        return classification

    def __add__(self, other):  # type: (ConfusionMatrix) -> ConfusionMatrix
        assert type(other) == ConfusionMatrix
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn
        )

    def __bool__(self):  # type: () -> bool
        # We need to do this explicitly because if I just return the if condition, we somehow end up with an int.
        # Yay, ducktyping!
        if self.tp and self.fp and self.fn and self.tn:
            return True
        else:
            return False

    def __repr__(self):
        return 'ConfusionMatrix(TP={}, FP={}, FN={}, TN={})'.format(self.tp, self.fp, self.fn, self.tn)


def find_xml_files(directory):  # type: (str) -> Dict[str, Union[bytes, str]]
    """ Returns all the XML files in a directory as a dict

    :param directory: directory to find XML files within
    :return: Dict of drug IDs to XML file names
    """
    files = {}  # type: Dict[str, Union[bytes, str]]
    for file in os.listdir(directory):
        if file.endswith('.xml'):
            files[file.replace('.xml', '')] = os.path.join(directory, file)
    return files


def maybe_extract_code(code):  # type: (str) -> str
    """ Some of the effects in the gold standard/training files are of the form "ID: description".

    This method extracts the code portion, so teams are not penalized for omitting or using different descriptions.
    Also strips trailing and leading whitespace

    If multiple effect codes are given, they will be split on semi-colons, and each code will be extracted

    :param code: mention code to possibly extract
    :return: code as-is if there is no effect description, None if code is none, or the code without the description
    """
    if not code:
        return code

def read_xml(file, allow_multiple_codes=False):
    # type: (Union[bytes, str], bool) -> Label
    """ Creates the XML file at the given path, returning a Label object

    :param file: path to XML file containing SPL annotations
    :param allow_multiple_codes: whether the XML file is allowed to specify multiple codes per each mention
    :return: Label object containing annotations parsed from given XML file
    """
    root = ElementTree.parse(file).getroot()
    assert root.tag == 'Label', 'Root is not Label: ' + root.tag
    label = Label(root.attrib['drug'])
    assert len(root) == 3, 'Expected 3 Children: ' + str(list(root))
    assert root[0].tag == 'Text', 'Expected \'Text\': ' + root[0].tag
    assert root[1].tag == 'Sentences', 'Expected \'Sentences\': ' + root[0].tag

    # Parse Sentences element
    for sentence in root[1]:
        assert sentence.tag == 'Sentence', 'Expected \'Sentence\': ' + sentence.tag
        sentence_id = sentence.attrib['id']
        label.sentences[sentence_id] = Sentence(sentence_id=sentence_id, text=sentence.find('SentenceText').text)

        for mention in sentence.findall('Mention'):
            mention_type = mention.attrib['type']

            if mention_type == 'Trigger':
                mention_code = None
            else:
                mention_code = mention.attrib['code']
                if mention_code != 'NO MAP':
                    codes = mention.attrib['code'].split()
                    if allow_multiple_codes:
                        mention_code = set(map(maybe_extract_code, codes))
                    else:
                        if len(codes) > 1:
                            mention_code = codes[0]
                        mention_code = maybe_extract_code(mention_code)

            label.mentions.append(
                Mention(sentence_id=sentence.attrib['id'].strip(),
                        mention_id=mention.attrib['id'].strip(),
                        mention_type=mention_type.strip(),
                        span=mention.attrib['span'].strip(),
                        code=mention_code,
                        mention_str=mention.attrib['str'].strip())
            )


    label.validate()

    return label


# Validates performance metrics, mainly just comparing the sections/text to make sure they're identical
def check_files_match(gold_label, guess_label):  # type: (Label, Label) -> bool
    """ Does simple sanity checks to ensure gold and guess labels match

    :param gold_label: Label object containing gold-standard annotations
    :param guess_label: Label object containing guess annotations
    :return:
    """
    is_valid = True

    if guess_label.drug != guess_label.drug:
        logger.error('Gold label drug %s did not match guess label drug %s', gold_label.drug, guess_label.drug)
        is_valid = False

    gold_num_sent = len(gold_label.sentences)
    guess_num_sent = len(guess_label.sentences)
    if gold_num_sent != guess_num_sent:
        logger.error('Different number of sentences in gold (%s) and guess (%s) files', gold_num_sent, guess_num_sent)
        is_valid = False

    for sentence_id, guess_sentence in guess_label.sentences.items():
        if sentence_id not in gold_label.sentences:
            logger.error('Different sentence ID in GUESS: %s', sentence_id)
            is_valid = False

        gold_sentence = gold_label.sentences.get(sentence_id)
        if guess_sentence != gold_sentence:
            logger.error('Different sentence text in guess and gold')
            logger.error("Guess sentence: \"%s\"", repr(guess_sentence))
            logger.error("Gold sentence:  \"%s\"", repr(gold_sentence))
            is_valid = False

    return is_valid


def print_f(name, matrix, primary=False):  # type: (str, ConfusionMatrix, bool) -> ()
    """ Prints various numbers related to F-measure

    :param name: Title to prin t
    :param matrix:
    :param primary:
    :return: None
    """
    print('  ' + name)
    print('    TP: {}  FP: {}  FN: {}'.format(matrix.tp, matrix.fp, matrix.fn))
    print('    Precision: {:.2f}'.format(matrix.precision))
    print('    Recall:    {:.2f}'.format(matrix.recall))
    print('    F1:        {:.2f}{}'.format(matrix.f1, '  (PRIMARY)' if primary else ''))


def print_macro_f(matrices, primary=True):  # type: (Sequence[ConfusionMatrix], bool) -> ()
    """ # Prints various numbers related to macro F-measure

    :param matrices: One or more ConfusionMatrices to compute macro F-measure from
    :param primary: Whether the Macro-F1 measure for this set of confusion matrices is the primary metric
    :return: None
    """
    merge = ConfusionMatrix.merge_confusion_matrices(matrices)
    length = len(matrices)
    print('    TP: {}  FP: {}  FN: {}'.format(merge.tp, merge.fp, merge.fn))
    print('    Micro-Precision: {:.2f}'.format(merge.precision))
    print('    Micro-Recall:    {:.2f}'.format(merge.recall))
    print('    Micro-F1:        {:.2f}'.format(merge.f1))
    print('    Macro-Precision  {:.2f}'.format(sum([c.precision for c in matrices]) / length))
    print('    Macro-Recall     {:.2f}'.format(sum([c.recall for c in matrices]) / length))
    print('    Macro-F1         {:.2f}{}'.format(sum([c.f1 for c in matrices]) / length,
                                                 '  (PRIMARY)' if primary else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TAC 2019 Drug Drug Interactions Extraction task')
    parser.add_argument('gold_dir', metavar='GOLD', type=str, help='path to directory containing system output')
    parser.add_argument('guess_dir', metavar='GUESS', type=str, help='path to directory containing system output')
    parser.add_argument('-1', '--task1', action='store_true', dest='task1', help='Evaluate Task 1')
    # parser.add_argument('-2', '--task2', action='store_true', dest='task2', help='Evaluate Task 2')
    # parser.add_argument('-3', '--task3', action='store_true', dest='task3', help='Evaluate Task 3')
    # parser.add_argument('-4', '--task4', action='store_true', dest='task4', help='Evaluate Task 4')
    parser.add_argument('--triggers', action='store_true', dest='evaluate_triggers',
                        help='Whether to evaluate triggers')
    args = parser.parse_args()

    logger.info('Evaluation script version: %s', VERSION)
    logger.info('Gold directory:  %s', args.gold_dir)
    logger.info('Guess directory: %s', args.guess_dir)

    tasks = [args.task1]
    if sum(tasks) == 0:
        args.task1 = args.task2 = args.task3 = args.task4 = True
    results = Results(args.task1, '', '', '', evaluate_triggers=args.evaluate_triggers)
    results.evaluate_dirs(args.gold_dir, args.guess_dir)

    for task in results:
        print('--------------------------------------------------')
        task.print_results()

