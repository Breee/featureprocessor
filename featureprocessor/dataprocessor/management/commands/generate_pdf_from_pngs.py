from django.core.management.base import BaseCommand
import os
from subprocess import Popen
import img2pdf



class Command(BaseCommand):
    help = 'read csv and create SMTfeatures.'

    def add_arguments(self, parser):
        # Positional arguments are standalone name
        # Positional arguments are standalone name
        parser.add_argument('images', nargs='+', default=[])
        parser.add_argument('dumppath')

    def handle(self, *args, **kwargs):
        # imagelist is the list with all image filenames
        imagelist = kwargs['images']
        dumppath = kwargs['dumppath']
        with open(f"{dumppath}/features.pdf", "wb") as f:
            # remove alpachannel from pngs
            for i in imagelist:
                os.popen(f'convert {i} -background white -alpha remove -alpha off {i}')
            # write pngs to pdf
            f.write(img2pdf.convert([i for i in imagelist if i.endswith(".png")]))