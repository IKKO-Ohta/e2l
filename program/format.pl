#! /usr/bin/perl
#
# Preprocessing script for training text of language models for Eco2Lang
# (Dow Jones English news articles)
#
# $Id$
#

use utf8;
use strict;


# read from stdin
while(<>){
    # remove white spaces at the beginning/end of sentence
    s/^\s+//;
    s/\s+$//;

    # empty line (=end of paragraph) -> add special symbol
    if( $_ eq "" ){
	$_ = "<PAR>";
	&Output($_);
	next;
    }
    
    # ignore header and footer lines
    if( /^MARKET SNAPSHOT/i || /^\(END\) Dow Jones/i ){ next; }
    # second line of header; often followed by main text
    s/^By .+ Marketwatch//i;

    # remove leading phrase like "SAN FRANCISCO (MarketWatch) --"
    s/^.+\(marketwatch\)\s+\-\-+//i;

    # skip if the line becomes empty by these processes
    if( $_ eq "" ){ next; }
    
    # convert to lower case
    $_ = lc($_);

    # remove (replace with space) quotation marks
    s/[\'\"]/ /g;

    # remove (replace with space) parentheses
    s/[\(\)\[\]\{\}\<\>]/ /g;

    # change punctuation marks to special symbols
    s/\.\s*$/ <PUNC>/;
    s/[\?\!\:\;]($|\s)/ <PUNC> /g;
    if(! /\<PUNC\>\s*$/ ){
	$_ .= " <PUNC>";
    }
    s/\,( |$)/ <COMMA> /g;

    # remove (replace with space) two or more successive hyphens
    s/\-\-+/ /g;
    
    # replace entity reference to actual character
    s/\&apos\;/'/g;

    # change numeric figures and percentages to special symbol
    s/(\d+)\%/$1 percent /g;
    s/\$\s*([\d\.\,]+)/dollar <NUM> /g;
    s/(^|\s)[\d\,]+($|[\s\!\?])/$1<NUM>$2/g;
    s/(^|\s)[\d\,]+\.\d*($|[\s\,\!\?])/$1<NUM>$2/g;
    s/(^|\s)\.\d+($|[\s\,\!\?])/$1<NUM>$2/g;

    # remove (replace with space) URLs
    s/(http|https)\:[^\s]+/ /g;
    
    # trim successive spaces
    s/\s+/ /g;

    # remove (again) white spaces at the beginning/end of sentence
    s/^\s+//;
    s/\s+$//;

    &Output($_);
}


# define how to provide processed text
sub Output {
    # simply show it to stdout
    print $_,"\n";

    return;
}
