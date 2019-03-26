import sys

## Method reads eclipse datafile and collects 
#  some information for the environment.
#  @param filename Path to the simulation datafile.
def read_data(filename):
    dimens = [0]*3
    skip_solution_kw = ['EQUIL']
    solution_section = []
    summary_section = []
    welspecs = []
    try:
        with open(filename, 'r') as fileobj:
            for line in fileobj:
                keyword = prepare_string(line)
                if len(keyword) > 0 and keyword[0][0].isalpha():
                    if keyword[0] == 'DIMENS':
                        temp = prepare_string(next(fileobj))
                        try:
                            dimens[0], dimens[1], dimens[2] = \
                                int(temp[0]), int(temp[1]), int(temp[2])
                        except ValueError:
                            print("### Error: Line after DIMENS keyword should consist of numeric data. Example: 60 60 7 /")
                            sys.exit(1)
                    if keyword[0] == 'SOLUTION':
                        keyword = parse_section(fileobj, 'SUMMARY', skip_solution_kw, \
                            True, solution_section)
                    if keyword[0] == 'SUMMARY':
                        keyword = parse_section(fileobj, 'SCHEDULE', [], False, summary_section)
                    if keyword[0] == 'WELSPECS':
                        keyword = parse_section(fileobj, '/', [], False, welspecs)
        fileobj.close()
    except:
        print("### Error while reading Eclipse data file")
        sys.exit(1)
    return dimens, solution_section, summary_section, welspecs

## Method that iterates inside a section
#  until it finds a stopping keyword. 
#  Returns last read keyword. 
#  @param file_obj The opened file object.
#  @param stopping_kw The stopping criteria.
#  @param skip_kw The list of keywords to skip.
#  @param check_alpha Check if the first element is not numeric.
#  @param ouput_list Parsing results.
def parse_section(file_obj, stopping_kw, skip_kw, check_alpha, ouput_list):
    keyword = ''
    for line in file_obj:
        keyword = prepare_string(line)
        if len(keyword) > 0:
            if keyword[0] == stopping_kw:
                break
            if keyword[0] in skip_kw:
                continue
            ouput_list.append(check_isalpha(keyword, check_alpha))
    return keyword

## Helper method that prepares a string line. 
def prepare_string(line):
    keyword = line.strip()
    keyword = keyword.split()
    return keyword
    
## Check if the first element is not numeric.
def check_isalpha(keyword, check):
    kw_out = ['']
    if check == False or (check == True and keyword[0][0].isalpha()):
        kw_out = keyword
    return kw_out