import glob
import os
import platform


def symlink(source, target, alias=None):
    if not os.path.exists(source):
        raise FileNotFoundError('SOURCE ({})'.format(source))

    if not os.path.exists(target):
        os.makedirs(target)

    def win_symlink(src, tgt):
        import winshell
        shortcut = winshell.shortcut(src)
        lnk_filepath = tgt + '.lnk'
        shortcut.write(lnk_filepath)
        return lnk_filepath

    def unix_symlink(src, dst):
        os.symlink(src, dst)
        return dst

    symlink_func =\
        win_symlink if platform.system() == 'Windows' else unix_symlink

    link_name = os.path.join(target, alias or os.path.basename(source))
    return symlink_func(source, link_name)
