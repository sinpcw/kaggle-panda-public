import os
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('input_csv', None, '入力CSVファイル名', short_name='i')
flags.DEFINE_string('output_csv', None, '出力CSVファイル名', short_name='o')

def main(argv):
    icsv = FLAGS.input_csv
    ocsv = FLAGS.output_csv
    if ocsv is None:
        dname = os.path.dirname(icsv)
        fname = os.path.basename(icsv)
        ocsv = os.path.join(dname, 'fix_{}'.format(fname))
    with open(icsv) as f:
        head = f.readline()
        body = f.readline()
    headitems = head.split(',')
    bodyitems = body.split(',')
    if len(headitems) == len(bodyitems):
        print('already fiexd')
    else:
        if headitems[-1].endswith('\r\n'):
            headitems[-1] = headitems[-1][:-2]
        elif headitems[-1].endswith('\n'):
            headitems[-1] = headitems[-1][:-1]
        print('count mismatch start fixed')
        tailcount = int(headitems[-1][5:])
        with open(icsv) as f:
            obj = f.readlines()
        with open(ocsv, mode='w') as f:
            s = headitems[0]
            for i in range(1, len(headitems)):
                s += ',{}'.format(headitems[i])
            s += ',feat_{}\n'.format(tailcount + 1)
            f.write(s)
            for itr in range(1, len(obj)):
                f.write(obj[itr])

if __name__ == '__main__':
    app.run(main)