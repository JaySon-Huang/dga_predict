"""Run experiments and create figs"""

import dga_classifier.bigram as bigram
import dga_classifier.lstm as lstm


def run_experiments(isbigram=True, islstm=True, nfolds=10):
    """Runs all experiments"""
    bigram_results = None
    lstm_results = None

    if isbigram:
        print("Running bigram...")
        bigram_results = bigram.run(nfolds=nfolds)
        print("====== bigram run finish ======")

    if islstm:
        print("Running lstm...")
        lstm_results = lstm.run(nfolds=nfolds)
        print("====== lstm run finish ======")

    return bigram_results, lstm_results


if __name__ == "__main__":
    run_experiments(isbigram=False, islstm=True, nfolds=1)  # Run with 1 to make it fast
