#import functools

def timer(func):
    #@functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        #print(f"Time elapsed: {end - start}")
        print("Funkcja {} wykonana w czasie: {}".format(func.__name__, round(end - start, 2)), "s")
        return value
    
    return wrapper