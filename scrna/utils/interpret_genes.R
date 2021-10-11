library(topGO)

loadings2go<-function(W,gene_2_GO,numtopgenes=100,min_genescore=1.0){
  #This blog was helpful: https://datacatz.wordpress.com/2018/01/19/gene-set-enrichment-analysis-with-topgo-part-1/
  #divide each gene by its rowmean to adjust for constant high expression
  W<-as.matrix(W/rowMeans(W)) #enables preservation of rownames when subsetting cols
  qtl<-1-(numtopgenes/nrow(W))
  topGenes<-function(x){
    cutoff<-max(quantile(x,qtl),min_genescore)
    x>cutoff
  }
  #W must be a matrix with gene names in rownames
  L<-ncol(W)
  res<-data.frame(dim=1:L,genes="",go_bp="")
  # gg<-names(tg[[12]])
  # gg<-gg[gg %in% names(gene_2_GO)]
  # geneList=factor(as.integer(bg %in% gg))
  for(l in 1:L){
    w<-W[,l]
    # names(geneList)<-rownames(W)
    tg<-head(sort(w,decreasing=TRUE),10)
    res[l,"genes"]<-paste(names(tg),collapse=", ")
    if(l==1){
      GOdata<-new('topGOdata', ontology='BP', allGenes=w, geneSel=topGenes, annot=annFUN.gene2GO, gene2GO=gene_2_GO, nodeSize=5)
    } else {
      GOdata<-updateGenes(GOdata,w,topGenes)
    }
    #wk<-runTest(GOdata,algorithm='weight01', statistic='ks')
    wf<-runTest(GOdata, algorithm='weight01', statistic='fisher')
    gotab=GenTable(GOdata,weightFisher=wf,orderBy='weightFisher',numChar=1000,topNodes=5)
    res[l,"go_bp"]<-paste(gotab$Term,collapse="; ")
  }
  res
}

jaccard<-function(a,b){
  i<-length(intersect(a,b))
  denom<-ifelse(i>0, length(a)+length(b)-i, 1)
  i/denom
}

# panglao_make_ref<-function(panglao_tsv="scrna/resources/PanglaoDB_markers_27_Mar_2020.tsv.gz"){
#   mk<-read.table(gzfile(panglao_tsv),header=TRUE,sep="\t")
#   tapply(mk$official.gene.symbol,mk$cell.type,function(x){x},simplify=FALSE)
# }

genes2celltype<-function(genelist,panglao_ref,gsplit=TRUE,ss=0){
  #genelist: a string with gene names separated by commas OR a list of genes
  #panglao_ref: a list whose names are cell types and values are lists of genes
  #gsplit: if TRUE, splits genelist into a list. If FALSE, assumes genelist already split
  #returns : cell type with highest jaccard similarity to the gene list
  g<-ifelse(gsplit, strsplit(genelist,", ",fixed=TRUE)[[1]], genelist)
  if(ss>0 && ss<length(g)){g<-g[1:ss]}
  g<-toupper(g)
  j<-vapply(panglao_ref,jaccard,FUN.VALUE=1.0,g) #jaccard scores for each cell type
  jmx<-max(j)
  if(jmx>0){
    res<-j[j==jmx]
    return(names(res))
  } else { #case where no cell types were found
    return(NA)
  }
}