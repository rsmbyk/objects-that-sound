import os
import platform
import subprocess


def symlink(source, target, alias=None):
    if not os.path.exists(source):
        raise FileNotFoundError('SOURCE ({})'.format(source))

    if not os.path.exists(target):
        os.makedirs(target)

    def win_symlink(src, tgt):
        if platform.win32_ver()[0] == '7':
            import winshell
            shortcut = winshell.shortcut(src)
            tgt = tgt + '.lnk'
            shortcut.write(tgt)
        elif not os.path.exists(tgt):
            subprocess.check_call(f'mklink /J \"{tgt}\" \"{src}\"', shell=True)
        return tgt

    def unix_symlink(src, dst):
        os.symlink(src, dst)
        return dst

    symlink_func =\
        win_symlink if platform.system() == 'Windows' else unix_symlink

    link_name = os.path.join(target, alias or os.path.basename(source))
    return symlink_func(source, link_name)
