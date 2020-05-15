
from os import chdir, makedirs, walk
from os.path import isdir, join
from shutil import copyfile
from sys import executable

from sdss_install.install import Install as InstallBase

class Install(InstallBase):

    def build(self, install_args=None):
        chdir(self.directory["work"])
        print("In Monkey patch!!")

        if install_args is not None:
            install_args = install_args.split(" ")
        else:
            install_args = []

        if 'python' in self.build_type:
            command = [executable,
                        'setup.py',
                        'install',
                    ]
            command.extend(install_args)
            #            "--prefix=%(install)s" % self.directory]
            self.logger.debug(' '.join(command))
            if not self.options.test:
                (out,err,proc_returncode) = self.execute_command(command=command)
                self.logger.debug(out)
                if proc_returncode != 0:
                    self.logger.error("Error during installation:")
                    self.logger.error(err)
                    self.ready = False
    #
            # Copy additional files
            #
            md = None
            cf = None
            if isdir('etc'):
                md = list()
                cf = list()
                for root, dirs, files in walk('etc'):
                    for d in dirs:
                        md.append(join(self.directory['install'],root,d))
                    for name in files:
                        if name.endswith('.module'):
                            continue
                        cf.append((join(root,name),
                                    join(self.directory['install'],
                                    root,
                                    name)))
            if md or cf:
                etc_dir = join(self.directory['install'],'etc')
                self.logger.debug('Creating {0}'.format(etc_dir))
                makedirs(etc_dir)
                if md:
                    for name in md:
                        self.logger.debug('Creating {0}'.format(name))
                        if not self.options.test:
                            makedirs(name)
                if cf:
                    for src,dst in cf:
                        self.logger.debug('Copying {0} -> {1}'
                            .format(src,dst))
                        if not self.options.test:
                            copyfile(src,dst)
