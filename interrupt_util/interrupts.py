import signal
import time

class GracefulInterruptHandler:
    def __init__(self):
        self.stop_requested = False
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.last_interrupt_time = 0

    def attach(self):
        signal.signal(signal.SIGINT, self.handler)

    def detach(self):
        signal.signal(signal.SIGINT, self.original_sigint)

    def handler(self, signum, frame):
        current_time = time.time()
        # If pressed twice within 2 seconds, raise KeyboardInterrupt to kill the whole script
        if current_time - self.last_interrupt_time < 2.0:
            print("\n      [Ctrl+C twice!] Hard exiting the entire script...")
            signal.signal(signal.SIGINT, self.original_sigint)
            raise KeyboardInterrupt
            
        self.stop_requested = True
        self.last_interrupt_time = current_time
        print("\n      [Ctrl+C] Early stopping current model. Press Ctrl+C again within 2 seconds to hard exit.")
