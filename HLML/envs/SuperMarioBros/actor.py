from time import sleep

from pykeyboard import PyKeyboard

class Actor():
    def __init__(self):
        # Get the keyboard
        self.keyboard = PyKeyboard()

        # The reset button
        self._reset = "r"

        # All the keys for the game configuration
        self._jump = "a"
        self._special = "s"
        self._up = self.keyboard.up_key
		self._down = self.keyboard.down_key
        self._left = self.keyboard.left_key
        self._right = self.keyboard.right_key

        # All the keys being used
        self._keys = [self._jump, self._special, self._up, self._down,
                      self._left, self._right]

        # All of the possible actions
        self._actions = {0 : [self._jump],
						 1 : [self._jump, self._left],
						 2 : [self._jump, self._right],
                         3 : [self._special],
						 4 : [self._special, self._left],
						 5 : [self._special, self._right],
						 6 : [self._special, self._jump],
						 7 : [self._special, self._jump, self._left],
						 8 : [self._special, self._jump, self._right],
                         9 : [self._up],
						 10 : [self._down],
                         11 : [self._left],
                         12 : [self._right]}

        # The number of possible actinos
        self.num_actions = len(self._actions)

    def perform_action(self, action):
        # Loop through all the keys
        for key in self._keys:
            # Press the key if it was given
            if(key in keys):
                self.keyboard.press_key(key)

            # Release it otherwise
            else: 
                self.keyboard.release_key(key)

    def reset(self):
		"""
        self.keyboard.press_key(self._reset)
        sleep(1e-2)
        self.keyboard.release_key(self._reset)
		"""
        self.release_keys()

    def release_keys(self):
        for key in self._keys:
            self.keyboard.release_key(key)
