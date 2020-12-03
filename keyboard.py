from pynput import keyboard
pressed = None

def on_press(key):
    global pressed
    try:
        pressed = key.char

    except AttributeError:
        if key != keyboard.Key.esc:
            pressed = key


def on_release(key):
    if key != keyboard.Key.enter:
        # Stop listener
        return False


def map_key():
    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    listener.stop()
    return pressed
