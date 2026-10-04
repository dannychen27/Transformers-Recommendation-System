import pandas as pd

from recommender.data.preprocessing import (
    build_user_sequences,
    make_test_examples,
    make_training_examples,
    make_validation_examples,
    split_sequence,
)


def test_build_user_sequences():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "movie_id": [10, 20, 30, 40],
            "rating": [5, 4, 3, 5],
            "timestamp": [30, 10, 20, 15],
        }
    )

    sequences = build_user_sequences(df)

    assert len(sequences) == 2
    assert sequences[1] == [20, 30, 10]
    assert sequences[2] == [40]

def test_split_sequence():
    sequence = [1, 2, 3, 4, 5]

    train, validation, test = split_sequence(sequence)

    assert train == [1, 2, 3]
    assert validation == 4
    assert test == 5

def test_make_training_examples():
    sequence = [10, 20, 30, 40]

    examples = make_training_examples(sequence)

    assert examples == [
        ([10], 20),
        ([10, 20], 30),
        ([10, 20, 30], 40),
    ]


def test_make_validation_examples():
    sequences = {
        1: [10, 20, 30, 40],
        2: [50, 60, 70, 80],
    }

    examples = make_validation_examples(sequences)

    assert examples == [
        ([10, 20], 30),
        ([50, 60], 70),
    ]

def test_make_test_examples():
    sequences = {
        1: [10, 20, 30, 40],
        2: [50, 60, 70, 80],
    }

    examples = make_test_examples(sequences)

    assert examples == [
        ([10, 20, 30], 40),
        ([50, 60, 70], 80),
    ]

def test_make_test_examples_includes_validation_interaction():
    sequences = {
        1: [10, 20, 30, 40, 50],
    }

    examples = make_test_examples(sequences)

    assert examples == [
        ([10, 20, 30, 40], 50),
    ]

