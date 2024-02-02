% A Latex template for designing the bachelor thesis at the undergraduate
% Computer Science programme of the Faculty of Mathematics, Natural Sciences
% and Information Technologies, University of Primorska.
%
% Author: Domen Šoberl
% Date: June 2023

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2022/07/26 Famnit bachelor thesis]
\LoadClass[a4paper,12pt]{report}

% Template options:
% - [slovene]   Use the Slovenian language (default).
% - [english]   Use the English language.
% - [cover]     Design the cover page.
% - [review]    Produce the version with extra line spacing.
% - [bibandref] Rename the Bibliography title to "Bibliography and references / Literatura in viri".
\DeclareOption{slovene}{\def\@langslo{}}
\DeclareOption{english}{\def\@langeng{}}
\DeclareOption{cover}{\def\@hascover{}}
\DeclareOption{review}{\def\@isdraft{}}
\DeclareOption{bibandref}{\def\@bibandref{}}
\ProcessOptions\relax

% Required packages
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{framed}
\RequirePackage{tocloft}
\RequirePackage{caption}
\RequirePackage{refcount}
\RequirePackage{tabularx}
\RequirePackage{enumitem}
\RequirePackage{rotating}
\RequirePackage[absolute]{textpos}
\RequirePackage[implicit=false]{hyperref}
\RequirePackage{colortbl}
\RequirePackage[numbib,notlof,notlot,nottoc]{tocbibind}

% Remove list item separation globally.
\setlist{noitemsep}

% Load the babel according to the used language.
\ifdefined\@langeng
    \RequirePackage[english]{babel}
\else
    \RequirePackage[slovene]{babel}
\fi

% A private macro to output a string in the used language.
% Usage: \@lang{Slovene}{English}
\newcommand{\@lang}[2]{
    \ifdefined\@langeng
        #2
    \else
        #1
    \fi
}

% Variables that store the information about the thesis.
% These are assigned by the user with the below macros.
\def\@titleslo{~}
\def\@titleeng{~}
\def\@titlehead{~}
\def\@authorname{~}
\def\@authorsurname{~}
\def\@authorshort{~}
\def\@studyprogram{~}
\def\@mentorslo{~}
\def\@mentoreng{~}
\def\@comentorslo{~}
\def\@comentoreng{~}
\def\@workingcomentorslo{~}
\def\@workingcomentoreng{~}
\def\@datemonth{~}
\def\@dateyear{~}
\def\@keywordsslo{~}
\def\@keywordseng{~}
\def\@abstractslo{~}
\def\@abstracteng{~}

% Macros exported by the template to set the thesis information.
\renewcommand{\title}[3]{\def\@titleslo{#1}\def\@titleeng{#2}\def\@titlehead{#3}}
\renewcommand{\author}[3]{\def\@authorname{#1}\def\@authorsurname{#2}\def\@authorshort{#3}}
\newcommand{\studyprogram}[1]{\def\@studyprogram{#1}}
\newcommand{\mentor}[2]{\def\@mentorslo{#1}\def\@mentoreng{#2}}
\newcommand{\comentor}[2]{\def\@hascomentor{}\def\@comentorslo{#1}\def\@comentoreng{#2}}
\newcommand{\workingcomentor}[2]{\def\@hasworkingcomentor{}\def\@workingcomentorslo{#1}\def\@workingcomentoreng{#2}}
\renewcommand{\date}[2]{\def\@datemonth{#1}\def\@dateyear{#2}}
\newcommand{\keywords}[2]{\def\@keywordsslo{#1}\def\@keywordseng{#2}}
\renewcommand{\abstract}[2]{\def\@abstractslo{#1}\def\@abstracteng{#2}}
\newcommand{\acknowledgments}[1]{\def\@acknowledgments{#1}}
\newcommand{\licensepage}[1]{\def\@licensepage{#1}}

% Counters used to automatize collecting certain information about the thesis.
\newcounter{counter@allpages}          % The number of all pages.
\newcounter{counter@contentpages}      % The number of content pages.
\newcounter{counter@figures}           % The number of figures.
\newcounter{counter@tables}            % The number of tables.
\newcounter{counter@appendixstartpage} % On which page the appendices start.
\newcounter{counter@appendixpages}     % The number of pages appendices take.
\newcounter{counter@appendices}        % The number of appendices.
\newcounter{counter@bibitems}          % The number of references.
\newcounter{counter@abbreviations}     % The number of abbreviations.

% Set the document margins.
\geometry{
    top=2.5cm,
    left=2.5cm,
    right=2.5cm,
    bottom=2.5cm,
    headheight=1.5cm
}

% Set the default line spacing. If draft option has been passed,
% the line spacing is later redefined for chapters.
\setstretch{1.25}

% Set the header and the footer. The footer is empty.
\pagenumbering{Roman}
\pagestyle{fancy}
\lhead{
    \fontsize{9}{12}\selectfont
    \@authorshort.~\@titlehead.\newline
    {Univerza na Primorskem, Fakulteta za matematiko, naravoslovje in informacijske tehnologije,~\@dateyear}
}
\rhead{\thepage}
\lfoot{}
\cfoot{}
\rfoot{}

% Redefine the entry prefix for the list of figures.
\renewcommand{\cftfigpresnum}{\figurename~}
\setlength{\cftfignumwidth}{5em}
\renewcommand{\cftfigaftersnum}{:}

% Redefine the entry prefix for the list of tables.
\renewcommand{\cfttabpresnum}{\tablename~}
\setlength{\cfttabnumwidth}{5em}
\renewcommand{\cfttabaftersnum}{:}

% Set the chapter title format.
\titleformat{\chapter}
    {\Large\bf}{\thechapter}{1em}{\thispagestyle{fancy}}
    \titlespacing*{\chapter}{0pt}{-\topskip}{*2}

% Set the section title format.
\titleformat{\section}
    {\large\bf}{\thesection}{1em}{\thispagestyle{fancy}}
    \titlespacing*{\section}{0pt}{1.5em}{1em}

% Redefine the figure and table caption, so it is left-aligned,
% a bit smaller and with some spacing above and below.
\captionsetup{
    %singlelinecheck=off,
    font=footnotesize,
    skip=5pt
}

% Start a new list to store abbreviations. It is formatted as
% a table of contents, but with page numbers turned off.
\newcommand{\listabbreviationsname}{List of Abbreviations}
\newlistof{abbreviations}{abbr}{\listabbreviationsname}
\cftpagenumbersoff{abbreviations}

% The exported macro for the user to add an abbreviation.
\newcommand{\abbreviation}[2]{
    \addcontentsline{abbr}{abbreviations}{\parbox{2.5cm}{#1}#2}
    \stepcounter{counter@abbreviations}
}

% Start a new list for appendices.
% It is a regular table of contents format.
\newcommand{\listappendicesname}{List of Appendices}
\newlistof{appendices}{apc}{\listappendicesname}

% The exported macro for the user to add an appendix.
% The first argument is the appendix name, and the
% second argument is the appendix title (description).
\renewcommand{\appendix}[2]{
    \ifdefined\@hasappendices
        % If this is not the first appendix, just start a new page.
        \clearpage
    \else
        % If this is the first appendix, add a title and determine
        % that this thesis has appendices. But before that, also
        % print out the bibliography page.
        \ifdefined\@bibandref
            \page@bibliography*
        \else
            \page@bibliography
        \fi
        \def\@hasappendices{}
        \addtocounter{counter@allpages}{\value{page}}
        \setcounter{counter@appendixstartpage}{\value{counter@allpages}}
        \clearpage
        \pagenumbering{gobble}
        \thispagestyle{empty}
        \pagestyle{empty}
    \fi
    \addcontentsline{apc}{appendices}{#1\hspace{2em}#2\vspace{0.5em}}
    {\Large\bf\noindent{#1}}
    \par\vspace{0.5em}
    \noindent{\large{#2}}
    \par\vspace{1em}
    \@afterindentfalse
    \@afterheading
    \stepcounter{counter@appendices}
}

% If the Slovene language is used, rename the titles of all lists.
\ifdefined\@langeng
\else
    \addto\captionsslovene{
        \renewcommand{\contentsname}{Kazalo vsebine}
        \renewcommand{\listtablename}{Kazalo tabel}
        \renewcommand{\listfigurename}{Kazalo slik in grafikonov}
        \renewcommand{\listappendicesname}{Kazalo prilog}
        \renewcommand{\listabbreviationsname}{Seznam kratic}
    }
\fi

% Format the titles of all lists.
\tocloftpagestyle{fancy}
\setlength{\cftbeforetoctitleskip}{0em}
\setlength{\cftaftertoctitleskip}{1em}
\setlength{\cftbeforelottitleskip}{0em}
\setlength{\cftafterlottitleskip}{1em}
\setlength{\cftbeforeloftitleskip}{0em}
\setlength{\cftafterloftitleskip}{1em}
\setlength{\cftbeforeapctitleskip}{0em}
\setlength{\cftafterapctitleskip}{3em}
\setlength{\cftbeforeabbrtitleskip}{0em}
\setlength{\cftafterabbrtitleskip}{3em}

% The macro to build the cover page.
\newcommand{\page@cover}{
    \fontfamily{ptm}\selectfont
    \clearpage
    \thispagestyle{empty}
    \newgeometry{
        top=3cm,
        left=3.5cm,
        right=3.5cm,
        bottom=2.5cm
    }
    \begin{center}
        \large
        UNIVERZA NA PRIMORSKEM\\
        FAKULTETA ZA MATEMATIKO, NARAVOSLOVJE IN\\
        INFORMACIJSKE TEHNOLOGIJE
        \par\vspace{7cm}
        \large
        ZAKLJUČNA NALOGA\\
        \ifdefined\@langeng
            (FINAL PROJECT PAPER)\\
        \fi
        \Large\MakeUppercase{\@titleslo}\\
        \ifdefined\@langeng
            \Large{(\MakeUppercase\@titleeng)}
        \fi
    \end{center}
    \par\vfill
    \begin{flushright}
        \large\MakeUppercase{\@authorname~\@authorsurname}
    \end{flushright}
    \restoregeometry
}

% The maketitle macro is redefined to build the Famnit style title page.
\renewcommand{\maketitle}{
    \clearpage
    \thispagestyle{empty}
    \newgeometry{
        top=3cm,
        left=3.5cm,
        right=3.5cm,
        bottom=2.5cm
    }
    \begin{center}
        \large
        UNIVERZA NA PRIMORSKEM\\
        FAKULTETA ZA MATEMATIKO, NARAVOSLOVJE IN\\
        INFORMACIJSKE TEHNOLOGIJE
        \par\vspace{5cm}
        \normalsize
        Zaključna naloga\\
        \ifdefined\@langeng
            (Final project paper)\\
        \fi
        \large\textbf\@titleslo\\
        (\@titleeng)
    \end{center}
    \par\vspace{5cm}
    \noindent
    Ime in priimek:~\@authorname~\@authorsurname\newline
    Študijski program:~\@studyprogram\newline
    Mentor:~\@mentorslo
    \ifdefined\@hascomentor
        \newline
        Somentor:~\@comentorslo
    \fi
    \ifdefined\@hasworkingcomentor
        \newline
        Delovni somentor:~\@workingcomentorslo
    \fi
    \par\vfill
    \begin{center}
        \large Koper, \@dateyear
    \end{center}
    \restoregeometry
}

% The macro that builds the information pages in Slovene and English.
\newcommand{\page@information}{
    % The Slovene version of the information page.
    \clearpage
    \pagenumbering{Roman}
    \setcounter{page}{2}
    \section*{Ključna dokumentacijska informacija}
    \begin{framed}
        \setstretch{2}
        \noindent\parbox{\linewidth}{\raggedright
            \vspace{0.2cm}
            \noindent
            Ime in PRIIMEK:~\@authorname~\MakeUppercase{\@authorsurname}\newline
            Naslov zaključne naloge:~{\@titleslo}\newline
            Kraj:~Koper\newline
            Leto:~\@dateyear
        }
        \par\vspace{1.5em}\noindent
        \parbox{\linewidth}{\raggedright
            \parbox{0.49\linewidth}{Število strani:~\ref{value@numpages}}
            \ifnum\getrefnumber{value@figures}>0
                \parbox{0.49\linewidth}{Število slik:~\ref{value@figures}}
            \fi
            \ifnum\getrefnumber{value@tables}>0
                \parbox{0.49\linewidth}{Število tabel:~\ref{value@tables}}
            \fi
            \ifnum\getrefnumber{value@appendices}>0
                \parbox{0.49\linewidth}{Število prilog:~\ref{value@appendices}}
                \parbox{0.49\linewidth}{Število strani prilog:~\ref{value@appendixpages}}
            \fi
            \parbox{0.49\linewidth}{Število referenc:~\ref{value@bibitems}}
        }
        \par\vspace{1.5em}\noindent
        \parbox{\linewidth}{\raggedright
            Mentor:~\@mentorslo\newline
            \ifdefined\@hascomentor
                Somentor:~\@comentorslo\newline
            \fi
            \ifdefined\@hasworkingcomentor
                Delovni somentor:~\@workingcomentorslo\newline
            \fi
            Ključne besede:~\@keywordsslo
        }
        \par\vspace{1.5em}
        \setstretch{1.25}
        \noindent
        {\bf Izvleček:}\newline
        \@abstractslo
    \end{framed}
    \par
    % The English version of the information page.
    \newpage
    \section*{Key document information}
    \begin{framed}
        \setstretch{2}
        \noindent\parbox{\linewidth}{\raggedright
            \vspace{0.2cm}
            \noindent
            Name and SURNAME:~\@authorname~\MakeUppercase{\@authorsurname}\newline
            Title of the final project paper:~{\@titleeng}\newline
            Place:~Koper\newline
            Year:~\@dateyear
        }
        \par\vspace{1.5em}\noindent
        \parbox{\linewidth}{\raggedright
            \parbox{0.49\linewidth}{Number of pages:~\ref{value@numpages}}
            \ifnum\getrefnumber{value@figures}>0
                \parbox{0.49\linewidth}{Number of figures:~\ref{value@figures}}
            \fi
            \ifnum\getrefnumber{value@tables}>0
                \parbox{0.49\linewidth}{Number of tables:~\ref{value@tables}}
            \fi
            \ifnum\getrefnumber{value@appendices}>0
                \parbox{0.49\linewidth}{Number of appendices:~\ref{value@appendices}}
                \parbox{0.49\linewidth}{Number of appendix pages:~\ref{value@appendixpages}}
            \fi
            \parbox{0.49\linewidth}{Number of references:~\ref{value@bibitems}}
        }
        \par\vspace{1.5em}\noindent
        \parbox{\linewidth}{\raggedright
            Mentor:~\@mentoreng\newline
            \ifdefined\@hascomentor
                Co-Mentor:~\@comentoreng\newline
            \fi
            \ifdefined\@hasworkingcomentor
                Working Co-Mentor:~\@workingcomentorslo\newline
            \fi
            Keywords:~\@keywordseng
        }
        \par\vspace{1.5em}
        \setstretch{1.25}
        \noindent
        {\bf Abstract:}\newline
        \@abstracteng
    \end{framed}
}

% The macro to build the acknowledgemnts page.
% If there is no acknowledgment, the macro has no effect.
\newcommand{\page@acknowledgments}{
    \ifdefined\@acknowledgments
        \clearpage
        \pagenumbering{gobble}
        \chapter*{\@lang{Zahvala}{Acknowledgments}}
        \par\bigskip
        {\@acknowledgments}
    \fi
}

% The macro to build the licenses page.
% If the licenses content is left empty, the macro has no effect.
\newcommand{\page@licenses}{
    \ifdefined\@licensepage
        \clearpage
        \addtocounter{counter@allpages}{1}
        \thispagestyle{empty}
        {\@licensepage}
    \fi
}

% The macro to build the table of contents page.
\newcommand{\page@contents}{
    \clearpage
    \tableofcontents
}

% The macro to build the list of tables page.
% If there are no tables, the macro has no effect.
\newcommand{\page@tables}{
    \ifdefined\r@value@tables
        \ifnum\getrefnumber{value@tables}>0
            \clearpage
            \listoftables
        \fi
    \fi
}

% The macro to build the list of figures page.
% If there are no figures, the macro has no effect.
\newcommand{\page@figures}{
    \ifdefined\r@value@figures
        \ifnum\getrefnumber{value@figures}>0
            \clearpage
            \listoffigures
        \fi
    \fi
}

% The macro to build the list of appendices page.
% If there are no appendices, the macro has no effect.
\newcommand{\page@appendices}{
    \ifnum\getrefnumber{value@appendices}>0
        \clearpage
        \renewcommand{\cftdot}{}
        \listofappendices
    \fi
}

% The macro to build the list of abbreviations page.
% If there are no abbreviations, the macro has no effect.
\newcommand{\page@abbreviations}{
    \ifnum\value{counter@abbreviations}>0
        \clearpage
        \renewcommand{\cftdot}{}
        \listofabbreviations
    \fi
}

% A macro that makes the transition from frontmatter to mainmatter.
% This separates the above information pages from the thesis chapters.
\newcommand{\@startchapters}{
    \addtocounter{counter@allpages}{\value{page}}
    \clearpage
    \pagenumbering{arabic}
    \thispagestyle{fancy}
    \pagestyle{fancy}
    % If draft, set a bigger line spacing.
    \ifdefined\@isdraft
        \setstretch{2.5}
    \fi
}

% The nonstarred macro to build the bibliography page.
% This version also prints out "and references" in the title.
\newcommand{\page@bibliographystar}{
    \ifdefined\@langeng
        \renewcommand{\bibname}{BIBLIOGRAPHY AND REFERENCES}
    \else
        \renewcommand{\bibname}{LITERATURA IN VIRI}
    \fi
    \IfFileExists{bibliography.bib}{
        \bibliographystyle{plain}
        \bibliography{bibliography}
    }{
        \chapter*{\bibname}
    }
}

% The starred macro to build the bibliography page.
% This version prints out just "Bibliography" in the title.
\newcommand{\page@bibliographynostar}{
    \ifdefined\@langeng
        \renewcommand{\bibname}{BIBLIOGRAPHY}
    \else
        \renewcommand{\bibname}{LITERATURA}
    \fi
    \IfFileExists{bibliography.bib}{
        \bibliographystyle{plain}
        \bibliography{bibliography}
    }{
        \chapter*{\bibname}
    }
}

% The bibliography is placed automatically at the end of the document.
% If there are appendices, the bibliography is placed before appendices.
\newcommand{\page@bibliography}{
    \@ifstar{\page@bibliographystar}{\page@bibliographynostar}
}

% The bibitem macro is redefined (extended) with the reference counter.
% This macro is used by bibtex.
\let\bibitem@original\bibitem\relax
\renewcommand{\bibitem}[1]{
    \bibitem@original{#1}
    \stepcounter{counter@bibitems}
}

% The figure macro is redefined (extended) with the figure counter.
\renewcommand{\thefigure}{\arabic{counter@figures}} 
\let\figure@original\figure\relax
\renewcommand{\figure}[1][]{
    \figure@original[#1]
    \stepcounter{counter@figures}
}

% The table macro is redefined (extended) with the table counter.
\renewcommand{\thetable}{\arabic{counter@tables}} 
\let\table@original\table\relax
\renewcommand{\table}[1][]{
    \table@original[#1]
    \stepcounter{counter@tables}
}

% The macro to indicate where the content ends.
\newcommand{\backmatter}{
   \setcounter{counter@contentpages}{\value{page}}
   \addtocounter{counter@contentpages}{-1}
   \def\@hasbackmatter{}
}

% At document begin execute all the macros that build information pages.
% The cover page is excluded from the the number of pages.
\AtBeginDocument{
    \ifdefined\@hascover
        \page@cover
    \fi
    
    \maketitle
    
    \page@licenses
    \page@acknowledgments
    \page@information
    \page@contents
    \page@tables
    \page@figures
    \page@appendices
    \page@abbreviations

    \@startchapters
}

% At document end save the values of all counters to the auxiliary file.
% These saved values will be used at the next compilation pass to insert the
% information on the numbers of the counted elements into the information page.
\AtEndDocument{
    % If the thesis has no appendices, print out the bibliography page.
    % If there are appendices, the bibliography page has already been
    % printed out before the first appendix.
    \ifdefined\@hasappendices
    \else
        \ifdefined\@bibandref
            \page@bibliography*
        \else
            \page@bibliography
        \fi
    \fi

    % Save the counter of all pages.
    \addtocounter{counter@allpages}{\value{page}}
    \refstepcounter{counter@allpages}
    \label{value@allpages}

    % Save the page numbers.
    \ifdefined\@hasbackmatter
        \refstepcounter{counter@contentpages}
    \else
        \refstepcounter{counter@allpages}
    \fi
    \label{value@numpages}

    % Save the counter of content pages.
    \refstepcounter{counter@contentpages}
    \label{value@contentpages}

    % Save the counter of figures.
    \addtocounter{counter@figures}{-1}
    \refstepcounter{counter@figures}
    \label{value@figures}

    % Save the counter of tables.
    \addtocounter{counter@tables}{-1}
    \refstepcounter{counter@tables}
    \label{value@tables}

    % Save the counter of the number of appendices.
    \addtocounter{counter@appendices}{-1}
    \refstepcounter{counter@appendices}
    \label{value@appendices}

    % Save the counter of appendix pages.
    \ifdefined\@hasappendices
        \setcounter{counter@appendixpages}{\numexpr\value{counter@allpages}-\value{counter@appendixstartpage}\relax}
    \else
        \setcounter{counter@appendixpages}{0}
    \fi
    \addtocounter{counter@appendixpages}{-1}
    \refstepcounter{counter@appendixpages}
    \label{value@appendixpages}

    % Save the counter of references.
    \addtocounter{counter@bibitems}{-1}
    \refstepcounter{counter@bibitems}
    \label{value@bibitems}

    \clearpage
}

\endinput