#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = None
        self.freqs = None
        self._load_rules_from_file(grammar_file)

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """

        # initialize dicts and open grammar file
        self.rules = dict()
        self.freqs = dict()
        fp = open(grammar_file, "r")

        for line in fp.readlines():
            # remove leading spaces and comments
            line = line.strip()
            if line.find("#") != -1:
                line = line[:line.find("#")]

            # only consider lines that start with a number 
            if len(line) != 0 and line[0].isdigit():
                tokens = line.split()

                # first token is frequency of the rule
                freq = float(tokens[0])

                # second token is LHS of the rule
                LHS = tokens[1]

                # remaining tokens is RHS of the rule
                RHS = tokens[2:]

                # add rule using LHS as key and RHS as value
                if LHS not in self.rules.keys():
                    self.rules[LHS] = []
                self.rules[LHS].append(RHS)

                # add frequency using LHS as key and frequency as value
                if LHS not in self.freqs.keys():
                    self.freqs[LHS] = []
                self.freqs[LHS].append(freq)

    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """

        # base cases
        if max_expansions == 0:
            return "... "
        if start_symbol not in self.rules.keys():
            return start_symbol
        
        # retrieve rules and frequencies using start_symbol as key
        curr_rules = self.rules[start_symbol]
        curr_freqs = self.freqs[start_symbol]

        # select random rule based on frequencies
        rand_rule = random.choices(curr_rules, curr_freqs)[0]

        # create random sample sentence
        curr_sample = ""
        for str in rand_rule:
            if str not in self.rules.keys():
                # if str not a key in rules, it is a terminal word so add to sample
                curr_sample += str + " "
            else:
                # else str is a key in rules, so recurse on str
                curr_sample += self.sample(derivation_tree, max_expansions - 1, str)

        # derivation tree includes start_symbols and parentheses
        if derivation_tree:
            return f"({start_symbol} {curr_sample})"
        else:
            return curr_sample


####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
