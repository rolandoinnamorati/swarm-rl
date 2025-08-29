\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{NobArticle}[2024/04/27 NobArticle Class]

\usepackage{etoolbox}
\newtoggle{unnumberedsections}
\settoggle{unnumberedsections}{false}
\DeclareOption{unnumberedsections}{\settoggle{unnumberedsections}{true}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[letterpaper,10pt,twoside]{article}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[dvipsnames]{xcolor}
\usepackage{subcaption}
\usepackage{blindtext}
\usepackage[toc,page]{appendix}

% Margins
\usepackage[
	top=2.5cm,
	bottom=2.5cm,
	left=2cm,
	right=2cm,
	footskip=1cm, % Space from the bottom margin to the baseline of the footer
	headsep=0.75cm, % Space from the top margin to the baseline of the header
	columnsep=20pt, % Space between text columns (in twocolumn mode)
]{geometry}

\usepackage{fancyhdr}
\usepackage[bottom, hang]{footmisc} % Force footnotes to the bottom
\pagestyle{fancy} \fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\setlength{\footnotemargin}{6pt} % Footnote space
\fancyhead[RO]{\small\textit{\runninghead}}
\fancyhead[LE]{\small\textit{\runninghead}}
\fancyfoot[RO]{\small\textbf{\thepage}}
\fancyfoot[LO]{\footnotesize\footertext}
\fancyfoot[LE]{\small\textbf{\thepage}}
\fancyfoot[RE]{\footnotesize\footertext}

% Fonts
\usepackage[T1]{fontenc}
\usepackage[tt=false, type1=true]{libertine}
\usepackage[varqu]{zi4}
\usepackage[libertine]{newtxmath}

% Bibliography
\usepackage[
	backend=biber,
    style=numeric,
	citestyle=authoryear,
	bibstyle=authoryear,
	sorting=nyt,
    natbib=true,
    uniquelist=false
]{biblatex}

\setlength\bibitemsep{1.5\itemsep}
\setlength\bibhang{0pt}
\renewcommand*{\bibfont}{\scriptsize}
\addbibresource{refs.bib} % Reference document

\ExecuteBibliographyOptions{maxcitenames=1}
\DeclareFieldFormat{citehyperref}{
  \DeclareFieldAlias{bibhyperref}{noformat}% Avoid nested links
  \bibhyperref{#1}}
\DeclareFieldFormat{textcitehyperref}{
  \DeclareFieldAlias{bibhyperref}{noformat}
  \bibhyperref{
    #1
    \ifbool{cbx:parens}
      {\bibcloseparen\global\boolfalse{cbx:parens}}
      {}}}
\savebibmacro{cite}
\savebibmacro{textcite}
\renewbibmacro*{cite}{
  \printtext[citehyperref]{
    \restorebibmacro{cite}
    \usebibmacro{cite}}}
\renewbibmacro*{textcite}{
  \ifboolexpr{
    ( not test {\iffieldundef{prenote}} and
      test {\ifnumequal{\value{citecount}}{1}} )
    or
    ( not test {\iffieldundef{postnote}} and
      test {\ifnumequal{\value{citecount}}{\value{citetotal}}} )
  }
    {\DeclareFieldAlias{textcitehyperref}{noformat}}
    {}
  \printtext[textcitehyperref]{
    \restorebibmacro{textcite}
    \usebibmacro{textcite}}}

% Tables
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{multirow}
\newcommand{\customtableformatting}{
    \renewcommand{\arraystretch}{1.1}\footnotesize
}
\AtBeginEnvironment{tabular}{\customtableformatting}
\AtBeginEnvironment{tabularx}{\customtableformatting}

% Captions
\usepackage{caption}
\captionsetup{skip=6pt} % Whitespace between figures/tables and the caption
\captionsetup{labelfont={bf,footnotesize}, textfont={footnotesize}}
\captionsetup[table]{skip=2pt,singlelinecheck=off,labelsep=newline}
\captionsetup[figure]{skip=7pt,singlelinecheck=off,labelsep=period}

\usepackage{enumitem}
\setlist{noitemsep}

\usepackage[bookmarks]{hyperref}

\newcommand\shade{85}\colorlet{nobcitecolor}{Aquamarine}

\hypersetup{
	colorlinks=true, % Whether to color the text of links
	urlcolor=black, % Color for \url and \href links
	linkcolor=nobcitecolor!\shade!black, % Color for \nameref links
	citecolor=nobcitecolor!\shade!black, % Color of reference citations
}

\usepackage{abstract}
\renewcommand*\abstractname{\vspace{5pt}\normalsize{Abstract}\hfill\vspace{2pt}}
\renewcommand{\abstractnamefont}{\bfseries\vspace{0.5\baselineskip}}
\renewcommand{\abstracttextfont}{\vspace{-0.5\baselineskip}}

\renewenvironment{abstract}
{\par\rule{\linewidth}{.5pt}\par%
{\bfseries \abstractname:}\par}
{\par\rule{\linewidth}{.5pt}}

\usepackage{titling}
\setlength{\droptitle}{-4\baselineskip}
\pretitle{\begin{flushleft}\Large\bfseries}
\posttitle{\end{flushleft}}
\preauthor{\begin{flushleft}\small}
\postauthor{\end{flushleft}}
\predate{\begin{flushleft}\footnotesize}
\postdate{\end{flushleft}}
\setlength{\thanksmarkwidth}{3pt}
\setlength{\thanksmargin}{-3pt}
\patchcmd{\maketitle}{plain}{empty}{}{}

\usepackage{titlesec}

\iftoggle{unnumberedsections} {
    \setcounter{secnumdepth}{0} }
{   \setcounter{secnumdepth}{3} }

\titleformat{\section}[block]{
    \normalsize\raggedright\bfseries
}{\thesection}{0.5em}{}[]

\titleformat{\subsection}[block]{
    \raggedright\bfseries\itshape
}{\thesubsection}{0.5em}{}[]

\titleformat{\subsubsection}[runin]{
    \bfseries
}{\thesubsubsection}{5pt}{}[]

\titlespacing*{\section}{0pt}{1\baselineskip}{3pt}
\titlespacing*{\subsection}{0pt}{1\baselineskip}{3pt}
\titlespacing*{\subsubsection}{0pt}{1\baselineskip}{8pt}

\newcommand{\runninghead}[1]{\renewcommand{\runninghead}{\footnotesize #1}}
\newcommand{\footertext}[1]{\renewcommand{\footertext}{\footnotesize #1}}

\newcommand{\todo}[1]{\textcolor{red}{TODO: #1.}}