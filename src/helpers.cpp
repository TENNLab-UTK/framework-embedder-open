#include "helpers.hpp"

IndentString::IndentString() {
    str = "";
    indent_spaces = 0;
}

IndentString::IndentString(const std::string &s, unsigned int spaces) {
    str = s;
    indent_spaces = spaces;
}

IndentString &IndentString::operator=(const std::string &s) {
    str = indent_lines("", s);

    return *this;
}

IndentString &IndentString::operator+=(const std::string &s) {
    str += indent_lines(str, s);

    return *this;
}

IndentString &IndentString::operator+=(IndentString &is) {
    str += indent_lines(str, is.str);

    return *this;
}

IndentString IndentString::operator+(const std::string &s) {
    return IndentString(str + indent_lines(str, s), indent_spaces);
}

IndentString IndentString::operator+(IndentString &is) {
    return IndentString(str + indent_lines(str, is.str), indent_spaces);
}

void IndentString::set_indent_spaces(unsigned int spaces) {
    indent_spaces = spaces;
}

void IndentString::add_indent_spaces(int spaces) { indent_spaces += spaces; }

std::string IndentString::get_str() { return str; }

void IndentString::append(const std::string &s) {
    str.append(indent_lines(str, s));
}

void IndentString::append(IndentString &is) {
    str.append(indent_lines(str, is.get_str()));
}

std::string IndentString::indent_lines(const std::string &preceding_str,
                                       const std::string &unindented_str) {
    std::string s;
    std::string indent_str;
    unsigned int i;
    int newline_pos;
    char c;
    bool line_empty;

    indent_str = std::string(indent_spaces, ' ');

    line_empty = true;
    newline_pos = -1;
    for (i = 0; i < unindented_str.size(); i++) {
        c = unindented_str[i];
        s += c;

        if (c != ' ' && c != '\n' && c != '\t' && c != '\r') {
            line_empty = false;
        }

        if (c == '\n') {
            if (!line_empty && newline_pos != -1) {
                s.insert(newline_pos, indent_str);
            }

            line_empty = true;
            newline_pos = s.size();
        }
    }

    if (!line_empty && newline_pos != -1) {
        s.insert(newline_pos, indent_str);
    }

    if (preceding_str.size() == 0 ||
        preceding_str[preceding_str.size() - 1] == '\n') {
        return indent_str + s;
    } else {
        return s;
    }
}
