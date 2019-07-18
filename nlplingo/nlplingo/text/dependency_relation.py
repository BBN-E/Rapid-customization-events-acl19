

class DependencyRelation(object):
    def __init__(self, dep_name, dep_direction, connecting_token_index):
        self.dep_name = dep_name
        self.dep_direction = dep_direction
        self.connecting_token_index = connecting_token_index

    @staticmethod
    def up():
        return 'UP'

    @staticmethod
    def down():
        return 'DOWN'

    @staticmethod
    def find_dep_paths_to_root(token, sentence):
        max_length = 5
        all_paths = []
        """:type: list[list[nlplingo.text.text_span.DependencyRelation]]"""
        current_path = []

        find_dep_paths_from_token(token, max_length, current_path, all_paths, sentence.tokens)
        return all_paths

    @staticmethod
    def find_dep_paths_from_token(token, max_length, current_path, all_paths, sent_tokens):
        """Recursively, through depth-first, gets all dep-paths from initial token, of a max_length
        :type token: nlplingo.text.text_span.Token
        :type max_length: int
        :type current_path: list[nlplingo.text.text_span.DependencyRelation]
        :type all_paths: list[list[nlplingo.text.text_span.DependencyRelation]]
        :type sent_tokens: list[nlplingo.text.text_span.Token]

        sent_tokens: tokens of the sentence
        Returns: list[nlplingo.text.text_span.DependencyRelation]
        """
        if len(current_path) >= max_length or (len(current_path) > 0 and current_path[-1].dep_name == 'ROOT'):
            all_paths.append(current_path)
            return

        for r in token.dep_relations:
            parent_indices_on_path = set(cp.parent_token_index for cp in current_path)
            if r.parent_token_index not in parent_indices_on_path:  # to prevent revisiting, prevent loops
                parent_token = sent_tokens[r.parent_token_index]
                find_dep_paths_from_token(parent_token, max_length, current_path + [r], all_paths, sent_tokens)
                # all_paths.append(current_path + [r])

    @staticmethod
    def find_connecting_path(path1, path2, tokens):
        """
        :type path1: list[nlplingo.text.text_span.DependencyRelation]
        :type path2: list[nlplingo.text.text_span.DependencyRelation]
        :type tokens: list[nlplingo.text.text_span.Token]
        """
        indices1 = set(r.parent_token_index for r in path1)
        indices2 = set(r.parent_token_index for r in path2)
        common_parent_index = indices1.intersection(indices2)

        ret = []
        for index in common_parent_index:
            current_path = []
            for i, dep_r in enumerate(path1):
                if dep_r.parent_token_index == index:
                    for r in path1[0:i + 1]:
                        current_path.append(
                            'u:{}:{}:{}'.format(r.dep_name, r.parent_token_index, tokens[r.parent_token_index].text))
            for i, dep_r in enumerate(path2):
                if dep_r.parent_token_index == index:
                    j = i
                    while (j >= 0):
                        current_path.append('d:{}:{}:{}'.format(path2[j].dep_name, path2[j].child_token_index,
                                                                tokens[path2[j].child_token_index].text))
                        j -= 1
            ret.append(current_path)
        return ret

    @staticmethod
    def find_shortest_dep_paths_between_tokens(token1, token2, tokens):
        """
        :type token1: nlplingo.text.text_span.Token
        :type token2: nlplingo.text.text_span.Token
        :type tokens: list[nlplingo.text.text_span.Token]
        """
        all_paths = []

        # check whether token2 is in token1's path to root
        for path in token1.dep_paths_to_root:
            for i, dep_r in enumerate(path):
                if dep_r.parent_token_index == token2.index_in_sentence:
                    current_path = []
                    for r in path[0:i + 1]:
                        current_path.append(
                            'u:{}:{}:{}'.format(r.dep_name, r.parent_token_index, tokens[r.parent_token_index].text))
                    all_paths.append(current_path)

        # check whether token1 is in token2's path to root
        for path in token2.dep_paths_to_root:
            for i, dep_r in enumerate(path):
                if dep_r.parent_token_index == token1.index_in_sentence:
                    current_path = []
                    j = i
                    while (j >= 0):
                        current_path.append('d:{}:{}:{}'.format(path[j].dep_name, path[j].child_token_index,
                                                                tokens[path[j].child_token_index].text))
                        j -= 1
                    all_paths.append(current_path)

        for path1 in token1.dep_paths_to_root:
            for path2 in token2.dep_paths_to_root:
                all_paths.extend(find_connecting_path(path1, path2, tokens))

        min_len = 99
        for path in all_paths:
            if len(path) < min_len:
                min_len = len(path)

        ret_paths = []
        for path in all_paths:
            if len(path) == min_len:
                ret_paths.append(path)
        return ret_paths
