#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the first trained model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the second trained model",
    )
    parser.add_argument(
        "prior_probability",
        type=float,
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.model1)
    lm2 = LanguageModel.load(args.model2)
    lm1_prior = args.prior_probability
    lm2_prior = 1 - lm1_prior

    # sanity check: both language models loaded for text categorization have the same vocabulary
    if lm1.vocab != lm2.vocab:
        raise ValueError(f"Language models do not have same vocabulary.")
    
    lm1_count = 0
    lm2_count = 0
    total = 0

    log.info("Per-file log-probabilities:")
    for file in args.test_files:
        total += 1
        # calculate log probability of file given language model (category) or P(file | category) and add log of prior probability or P(category)
        # adding log of probabilities is equivalent to multiplying probabilities
        # Application of Bayes' rule: P(category | file) = P(file | category) * P(category), log of P(file | category) found from file_log_prob function, P(category) is the prior probability
        # Note: in Bayes' rule we did not divide by P(file) because this is the same for both categories
        lm1_log_prob: float = file_log_prob(file, lm1) + math.log(lm1_prior)
        lm2_log_prob: float = file_log_prob(file, lm2) + math.log(lm2_prior)
        # the greater the log probability, the higher the probability
        if lm1_log_prob >= lm2_log_prob:
            lm1_count += 1
            print(f"{args.model1}\t{file}")
        else:
            lm2_count += 1
            print(f"{args.model2}\t{file}")
    
    print(f"{lm1_count} files were more probably {args.model1} ({lm1_count / total * 100:.2f}%)")
    print(f"{lm2_count} files were more probably {args.model2} ({lm2_count / total * 100:.2f}%)")


if __name__ == "__main__":
    main()
