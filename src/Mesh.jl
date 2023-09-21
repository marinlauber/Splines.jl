# mesh module
mutable struct Mesh
    elemVertex::Array{Float64,2}
    elemNode::Array{Array{Int64,1},1}
    degP::Array{Int64,1}
    C::Array{Array{Float64,2},1}
    numBasis::Int64
    numElem::Int64
    controlPoints::Array{Float64, 2}
    weights::Array{Float64, 1}
    knots :: Array{Float64,1}
end

function Mesh1D(ptLeft, ptRight, numElem, degP)
    nrb = nrbline(ptLeft, ptRight)
    new_knots = collect(LinRange(ptLeft, ptRight, numElem+1)[2:end-1])
    nrb = nrbdegelev(nrb, [degP-1])
    nrb = nrbkntins(nrb, [new_knots])
    IGAmesh = genMesh(nrb)
    gauss_rule = genGaussLegendre(degP+1)
    return IGAmesh, gauss_rule
end

"""
create the IEN (node index to element) array for a given knot vector and p and
elementVertex array

INPUT: knotVector - the given knot vector
        numElem - number of non-zero knot-spans
        p - polynomial degree
OUTPUT: IEN - array with nb rows and p+1 columns, where each row indicates the
                global node indices for a knot-span
        elemVertex - array with nb rows and 2 columns, where each row indicates
        the left and right knot of a non-empty knot-span
"""
function makeIEN(knotVector, numElem, p)
    IEN  = zeros(Int64, numElem, p+1)
    tolEq = 1e-10
    elemVertex = zeros(numElem, 2)
    elementCounter = 0
    for indexKnot = 1:length(knotVector)-1
        if knotVector[indexKnot+1]>knotVector[indexKnot]+tolEq
            elementCounter += 1
            IEN[elementCounter,:] = indexKnot-p:indexKnot
            elemVertex[elementCounter, :] = [knotVector[indexKnot], knotVector[indexKnot+1]]
        end
    end
    @assert numElem==elementCounter "Wrong number of elements passed"
    return IEN, elemVertex
end

"""
Initializes an IGA mesh from a NURBS object
"""
function genMesh(nurbs::NURBS)
    #1D mesh
    knotU = nurbs.knots[1]
    degP = nurbs.order.-1
    Cmat, numElem = bezierExtraction(knotU, degP[1])
    C = Array{Array{Float64,2},1}(undef, numElem)
    for i=1:numElem
        C[i] = Cmat[:,:,i]
    end
    numBasis = length(knotU) - degP[1] - 1
    IEN, elemVertex = makeIEN(knotU, numElem, degP[1])
    cpts = reshape(nurbs.coefs[1:3,:,:,:], 3, numBasis)
    wgts = reshape(nurbs.coefs[4,:,:,:], numBasis)
    for i=1:3
        cpts[i,:] ./= wgts
    end
    elemNode = Array{Array{Int64,1},1}(undef, numElem)
    for i=1:numElem
        elemNode[i]=IEN[i,:]
    end
    IGAmesh = Mesh(elemVertex, elemNode, degP, C, numBasis, numElem, cpts, wgts, knotU)
    return IGAmesh
end
"""
Plots the basis functions of a mesh in parameter space
"""
function plotBasisParam(mesh::Mesh)
    #1D plot
    colorList = ["blue", "red", "green", "black", "magenta"]
    graph=Plots.plot(title="B-Splines of degree $(mesh.degP[1])")
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB, ddB = bernsteinBasis(evalPts, mesh.degP[1])

    for iBasis = 1:mesh.numBasis
        colorIndex = ((iBasis-1) % length(colorList))+1
        for iElem = 1:mesh.numElem
            localIndex = findall(isequal(iBasis), mesh.elemNode[iElem])
            if length(localIndex)>0
                uMin = mesh.elemVertex[iElem, 1]
                uMax = mesh.elemVertex[iElem, 2]
                plotPts = LinRange(uMin, uMax, numPtsElem)
                plotVal = B*(mesh.C[iElem][localIndex,:])'
                graph = plot!(plotPts, plotVal, color=colorList[colorIndex], leg=false, line=2)
                graph = scatter!([uMin], [0], color="red", leg=false, markersize=5)
                graph = scatter!([uMax], [0], color="red", leg=false, markersize=5)
            end
        end
    end
    display(graph)
end

"""
Plots the computed solution
"""
function plotSol(mesh::Mesh, sol0, exactSol::Function)
    graph=plot(title="Approximate solution")
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB = bernsteinBasis(evalPts, degP[1])

    for iElem in 1:mesh.numElem
        curNodes = mesh.elemNode[iElem]
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        plotPts = LinRange(uMin, uMax, numPtsElem)
        physPts = zeros(numPtsElem)
        splineVal = B*(mesh.C[iElem])'
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]
        basisVal = zero(splineVal)
        for iPlotPt = 1:numPtsElem
            RR = splineVal[iPlotPt,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum
            basisVal[iPlotPt,:] = RR
            physPts[iPlotPt] = RR'*cpts
        end
        solVal = basisVal*sol0[curNodes]
        graph = plot!(plotPts, solVal, color="blue", leg=false, line=2)
        exSolVal = real(exactSol.(physPts))
        graph = plot!(plotPts, exSolVal, color="green", leg=false, line=2)
        graph = plot!(plotPts, zeros(length(plotPts)), color="black", leg=false, line=1)
        graph = scatter!([uMin], [0], color="red", leg=false, markersize=5)
        graph = scatter!([uMax], [0], color="red", leg=false, markersize=5)
    end
    display(graph)
end


"""
Plots the error in the approximate solution
"""
function plotSolError(mesh::Mesh, sol0, exactSol::Function)
    graph=plot(title="Error \$u-u_h\$")
    numPtsElem = 11
    evalPts = LinRange(-1, 1, numPtsElem)
    B, dB = bernsteinBasis(evalPts, degP[1])

    for iElem in 1:mesh.numElem
        curNodes = mesh.elemNode[iElem]
        uMin = mesh.elemVertex[iElem, 1]
        uMax = mesh.elemVertex[iElem, 2]
        plotPts = LinRange(uMin, uMax, numPtsElem)
        physPts = zeros(numPtsElem)
        splineVal = B*(mesh.C[iElem])'
        cpts = mesh.controlPoints[1, curNodes]
        wgts = mesh.weights[curNodes]
        basisVal = zero(splineVal)
        for iPlotPt = 1:numPtsElem
            RR = splineVal[iPlotPt,:].* wgts
            w_sum = sum(RR)
            RR /= w_sum
            physPts[iPlotPt] = RR'*cpts
            basisVal[iPlotPt,:] = RR
        end
        exSolVal = real(exactSol.(physPts))
        solVal = basisVal*sol0[curNodes]
        graph = plot!(plotPts, exSolVal-solVal, color="blue", leg=false, line=2)
        graph = plot!(plotPts, zeros(length(plotPts)), color="black", leg=false, line=1)
        graph = scatter!([uMin], [0], color="red", leg=false, markersize=5)
        graph = scatter!([uMax], [0], color="red", leg=false, markersize=5)
    end
    display(graph)
end
