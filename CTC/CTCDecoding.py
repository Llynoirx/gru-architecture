import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        for t in range(len(y_probs[0])):
            max_prob = y_probs[0][t]
            max_idx = 0
            for sym in range(1, len(y_probs)):
                if max_prob < y_probs[sym][t]:
                    max_prob = y_probs[sym][t]
                    max_idx = sym
            path_prob *= max_prob

            if max_idx != 0:
            	decoded_path.append(self.symbol_set[max_idx-1])
            else:
                decoded_path.append(blank)

        compressed_path = [decoded_path[0]] + [decoded_path[i] for i in range(1, len(decoded_path)) 
                                               if decoded_path[i-1] != decoded_path[i] and decoded_path[i] != 0]
        decoded_path = ''.join(map(str, compressed_path))

        return decoded_path, path_prob



class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        path_score, blank_score = {}, {}
        _, seq_len, _ = y_probs.shape

        new_paths_final_blank, new_paths_final_sym, new_blank_score, new_path_score = self.init_paths(self.symbol_set, y_probs[:, 0, :])

        for t in range(1, seq_len):
            paths_final_blank, paths_final_sym, blank_score, path_score = self.prune(new_paths_final_blank, new_paths_final_sym, new_blank_score, new_path_score, self.beam_width)
            new_paths_final_blank, new_blank_score = self.extend_blank(paths_final_blank, paths_final_sym, blank_score, path_score, y_probs[:, t, :])
            new_paths_final_sym, new_path_score = self.extend_sym(paths_final_blank, paths_final_sym, blank_score, path_score, self.symbol_set, y_probs[:, t, :])

        _, final_path_score = self.merge_paths(new_paths_final_blank, new_paths_final_sym, new_blank_score, new_path_score)
        best_path = max(final_path_score, key=lambda x: final_path_score[x])

        return best_path, final_path_score

    

   
    def init_paths(self, symbol_set, y):
        paths_blank, paths_symbol = [""], []
        score_blank, score_path = {"" : y[0]}, {}

        for i, sym in enumerate(self.symbol_set):
            paths_symbol.append(sym)
            score_path[sym] = y[i + 1]

        return paths_blank, paths_symbol, score_blank, score_path

    def extend_blank(self, paths_final_blank, paths_final_sym, blank_score, path_score, y):
        updated_blank_score = {}
        updated_paths_final_blank = []

        for path in paths_final_blank:
            updated_paths_final_blank.append(path)
            updated_blank_score[path] = blank_score[path] * y[0]

        for path in paths_final_sym:
            if path in updated_paths_final_blank:
                updated_blank_score[path] += path_score[path] * y[0]
            else:
                updated_paths_final_blank.append(path)
                updated_blank_score[path] = path_score[path] * y[0]

        return updated_paths_final_blank, updated_blank_score

    def extend_sym(self, paths_final_blank, paths_final_sym, blank_score, path_score, symbol_set, y):
        updated_paths_final_sym = []
        updated_path_score = {}

        for path in paths_final_blank:
            for i, sym in enumerate(symbol_set):
                new_path = path + sym
                updated_paths_final_sym.append(new_path)
                updated_path_score[new_path] = blank_score[path] * y[i+1]

        for path in paths_final_sym:
            for i, sym in enumerate(symbol_set):
                new_path = path if sym == path[-1] else path + sym
                if new_path in updated_paths_final_sym:
                    updated_path_score[new_path] += path_score[path] * y[i+1]
                else:
                    updated_paths_final_sym.append(new_path)
                    updated_path_score[new_path] = path_score[path] * y[i+1]

        return updated_paths_final_sym, updated_path_score
    

    def merge_paths(self, paths_final_blank, paths_final_sym, blank_score, path_score):
        merged_paths = paths_final_sym
        final_path_score = path_score

        for p in paths_final_blank:
            if p in merged_paths:
                final_path_score[p] += blank_score[p]
            else:
                merged_paths.append(p)
                final_path_score[p] = blank_score[p]

        return merged_paths, final_path_score

    def prune(self, paths_final_blank, paths_final_sym, blank_score, path_score, beam_width):
        pruned_blank_score, pruned_path_score = {}, {}
        scores = [blank_score[p] for p in paths_final_blank] + [path_score[p] for p in paths_final_sym]
        scores = sorted(scores, reverse=True)

        cutoff = scores[beam_width] if beam_width < len(scores) else scores[-1]
        pruned_paths_final_blank, pruned_paths_final_sym = [], []

        for p in paths_final_blank:
            if blank_score[p] > cutoff:
                pruned_paths_final_blank.append(p)
                pruned_blank_score[p] = blank_score[p]

        for p in paths_final_sym:
            if path_score[p] > cutoff:
                pruned_paths_final_sym.append(p)
                pruned_path_score[p] = path_score[p]

        return pruned_paths_final_blank, pruned_paths_final_sym, pruned_blank_score, pruned_path_score


    