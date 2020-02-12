import threading


def fork(count, target, *args):
    if len(args) != count:
        raise ValueError(f'Number of arguments {len(args)} must be the same as number of threads {count}')

    threads = list()

    for idx in range(count):
        thread = threading.Thread(target=target, args=args[0])
        threads.append(thread)
        thread.start()

    for idx, thread in enumerate(threads):
        thread.join()
