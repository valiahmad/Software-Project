
# print('\x1b[2;30;42m' + 'TEXT' + '\x1b[0m')
            #[0-7Style;30-37Fore;40-47Back

SIMP = '\x1b[0;'
BOLD = '\x1b[1;'
ITALIC = '\x1b[3;'
UNDERLINE = '\x1b[4;'

fgray = '30;'
fred = '31;'
fgreen_yashmi = '32;'
forange = '33;'
fblue = '34;'
fpink = '35;'
fgreen = '36;'
fwhite = '37;'

bgray = '40m'
bred = '41m'
bgreen_yashmi = '42m'
borange = '43m'
bblue = '44m'
bpink = '45m'
bgreen = '46m'
bwhite = '47m'

End = '\x1b[0m'
ENDC = '\033[0m'

# print(SIMP+fblue+bgray+'\n Excel File Has Been Loaded!'+End)

# def print_format_table():

#     for style in range(8):

#         for fg in range(30,38):
#             s1 = ''
#             for bg in range(40,48):
#                 format = ';'.join([str(style), str(fg), str(bg)])
#                 s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
#             print(s1)
#         print('\n')

# print_format_table()
