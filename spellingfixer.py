import string
import math

def load_data(filename="aspell.txt") -> list[tuple[str, str]]:
    """
    Load observed (correct, typed) word pairs from a file.

    Args:
        filename (str, optional): defaults to "aspell.txt"

    Returns:
        list[tuple[str, str]]: list of (correct, typed) word pairs
    """
    pairs = []
    with open(filename, "r") as f:
        for line in f:
            if ":" in line:
                parts = line.strip().split(":", 1)
                correct = parts[0].strip().lower()
                wrongs = parts[1].split()
                for w in wrongs:
                    pairs.append((correct, w.strip().lower()))
    return pairs

def compute_emissions(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """
    Compute emission probabilities from observed pairs.

    Args:
        pairs (list[tuple[str, str]]): list of (correct, typed) word pairs

    Returns:
        dict[str, dict[str, float]]
    """
    emissions = {}

    for correct, typed in pairs:
        length = len(correct)
        if len(typed) < length:
            length = len(typed)

        for i in range(length):
            c_char = correct[i]
            t_char = typed[i]

            if c_char not in emissions:
                emissions[c_char] = {}
            if t_char not in emissions[c_char]:
                emissions[c_char][t_char] = 0
            emissions[c_char][t_char] += 1

    emission_probs = {}
    for c_char in emissions:
        total = 0
        for t_char in emissions[c_char]:
            total += emissions[c_char][t_char]
        emission_probs[c_char] = {}
        for t_char in emissions[c_char]:
            emission_probs[c_char][t_char] = emissions[c_char][t_char] / total

    return emission_probs

def compute_transitions(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """
    Compute character transition probabilities from correct words.

    Args:
        pairs (list[tuple[str, str]]): list of (correct, typed) pairs

    Returns:
        dict[str, dict[str, float]]: transition_probs[a][b] = P(b | a)
    """
    transitions = {}

    for correct, _ in pairs:
        word = "^" + correct + "$"
        for i in range(len(word) - 1):
            a = word[i]
            b = word[i + 1]

            if a not in transitions:
                transitions[a] = {}
            if b not in transitions[a]:
                transitions[a][b] = 0
            transitions[a][b] += 1

    transition_probs = {}
    for a in transitions:
        total = 0
        for b in transitions[a]:
            total += transitions[a][b]
        transition_probs[a] = {}
        for b in transitions[a]:
            transition_probs[a][b] = transitions[a][b] / total

    return transition_probs


def viterbi(typed, transition_probs, emission_probs):
    """
    Viterbi algorithm to find the most likely correct word for an observed word.

    Args:
        typed (str): the observed word
        transition_probs (dict): mapping as returned by compute_transitions()
        emission_probs (dict): mapping as returned by compute_transitions()

    Returns:
        str: the most likely correct word
    """
    letters = list(string.ascii_lowercase)
    V = [{}]
    path = {}

    for s in letters:
        trans_prob = transition_probs.get("^", {}).get(s, 1e-6)
        emit_prob = emission_probs.get(s, {}).get(typed[0], 1e-6)
        V[0][s] = math.log(trans_prob) + math.log(emit_prob)
        path[s] = [s]

    for t in range(1, len(typed)):
        V.append({})
        newpath = {}
        for s in letters:
            max_prob = float("-inf")
            prev_state = None

            for s0 in letters:
                prev_prob = V[t - 1].get(s0, float("-inf"))
                trans_prob = transition_probs.get(s0, {}).get(s, 1e-6)
                emit_prob = emission_probs.get(s, {}).get(typed[t], 1e-6)
                prob = prev_prob + math.log(trans_prob) + math.log(emit_prob)

                if prob > max_prob:
                    max_prob = prob
                    prev_state = s0

            V[t][s] = max_prob
            if prev_state is not None:
                newpath[s] = path[prev_state] + [s]

        path = newpath

    max_prob = float("-inf")
    best_state = None
    for s in letters:
        last_prob = V[len(typed) - 1].get(s, float("-inf"))
        end_prob = math.log(transition_probs.get(s, {}).get("$", 1e-6))
        total_prob = last_prob + end_prob
        if total_prob > max_prob:
            max_prob = total_prob
            best_state = s

    if best_state is None:
        return typed
    return "".join(path[best_state])


def closest_known_word(candidate: str, dictionary_words: list[str]) -> str:
    """
    Map a candidate word to the closest word in a known dictionary.

    Args:
        candidate (str): the word to match
        dictionary_words (list[str]): list of valid dictionary words

    Returns:
        str: the closest matching word from the dictionary, or the candidate itself if it is already in the dictionary
    """
    if candidate in dictionary_words:
        return candidate

    best = candidate
    best_score = 10**9

    for word in dictionary_words:
        dist = abs(len(word) - len(candidate))
        m = min(len(word), len(candidate))
        for i in range(m):
            if word[i] != candidate[i]:
                dist += 1
        if dist < best_score:
            best_score = dist
            best = word
    return best


def main():
    print("Loading data and training model...")
    pairs = load_data("aspell.txt")
    emission_probs = compute_emissions(pairs)
    transition_probs = compute_transitions(pairs)

    dictionary_words = []
    for correct, _ in pairs:
        if correct not in dictionary_words:
            dictionary_words.append(correct)

    print("Model ready. Type something (blank line to quit).")

    while True:
        text = input("\nEnter text: ").strip()
        if not text:
            break

        corrected = []
        for word in text.split():
            decoded = viterbi(word.lower(), transition_probs, emission_probs)
            fixed = closest_known_word(decoded, dictionary_words)
            corrected.append(fixed)

        print("Corrected:", " ".join(corrected))


if __name__ == "__main__":
    main()