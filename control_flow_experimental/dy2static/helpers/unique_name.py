import collections


class UniqueNameGenerator:
    """
    Generate unique name with prefix.

    Args:
        prefix(str): The generated name prefix. All generated name will be
                     started with this prefix.
    """

    def __init__(self, prefix=None):
        self.ids = collections.defaultdict(int)
        if prefix is None:
            prefix = ""
        self.prefix = prefix

    def __call__(self, key):
        """
        Generate unique names with prefix

        Args:
            key(str): The key of return string.

        Returns(str): A unique string with the prefix
        """
        tmp = self.ids[key]
        self.ids[key] += 1
        return self.prefix + "_".join([key, str(tmp)])


generator = UniqueNameGenerator()


def generate(key):
    """
    Generate unique name with prefix key. Currently, Paddle distinguishes the
    names of the same key by numbering it from zero. For example, when key=fc,
    it continuously generates fc_0, fc_1, fc_2, etc.

    Args:
        key(str): The prefix of generated name.

    Returns:
        str: A unique string with the prefix key.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> name1 = paddle.utils.unique_name.generate('fc')
            >>> name2 = paddle.utils.unique_name.generate('fc')
            >>> print(name1, name2)
            fc_0 fc_1
    """
    return generator(key)

