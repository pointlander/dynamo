package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/goml/gobrain"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	//"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/kortschak/nmf"
	"github.com/pointlander/kmeans"
	"github.com/skelterjohn/go.matrix"
)

const (
	NEURAL_WIDTH  = 4
	NEURAL_MIDDLE = 3
	MODEL_WIDTH   = (NEURAL_WIDTH+1)*(NEURAL_MIDDLE+1) + (NEURAL_MIDDLE+1)*NEURAL_WIDTH
	NMF_WIDTH     = 2
	SAMPLES       = 4
)

type Vectorizer struct {
	Abbr, Name string
	Vectorize  func(prices []Quote, minYear, maxYear int) *matrix.DenseMatrix
}

type Vector []float64

/* https://en.wikipedia.org/wiki/Cosine_similarity */
func (a Vector) similarity(b Vector) float64 {
	cols := len(a)
	dot, maga, magb := float64(0), float64(0), float64(0)
	for j := 0; j < cols; j++ {
		aij, bij := a[j], b[j]
		dot += aij * bij
		maga += aij * aij
		magb += bij * bij
	}
	mag := math.Sqrt(maga * magb)
	if mag == 0 {
		return 0
	}
	return dot / mag
}

func convert(item []string) (time.Time, float64) {
	t, err := time.Parse("2006-01-02", item[0])
	if err != nil {
		log.Fatal(err)
	}
	p, err := strconv.ParseFloat(item[1], 64)
	if err != nil {
		log.Fatal(err)
	}
	return t, p
}

func iterator(items [][]string, iter func(index int, t time.Time, price float64)) {
	i := 0
	x1, y1 := convert(items[0])
	iter(i, x1, y1)
	i++
	for _, item := range items[1:] {
		x2, y2 := convert(item)
		hours := x1.Sub(x2).Hours()
		if hours > 24 {
			s := (y1 - y2) / hours
			for x := hours - 24; x > 0; x -= 24 {
				y := s*x + y2
				iter(i, x2.Add(time.Duration(x)*time.Hour), y)
				i++
			}
		}
		iter(i, x2, y2)
		i++
		x1, y1 = x2, y2
	}
}

type Quote struct {
	t     time.Time
	price float64
}

func norm(x *matrix.DenseMatrix) {
	rows, cols := x.Rows(), x.Cols()
	for i := 0; i < rows; i++ {
		sum := float64(0)
		for j := 0; j < cols; j++ {
			k := x.Get(i, j)
			sum += k * k
		}
		sum = math.Sqrt(sum)
		if sum == 0 {
			continue
		}
		for j := 0; j < cols; j++ {
			x.Set(i, j, x.Get(i, j)/sum)
		}
	}
}

func subtract(x *matrix.DenseMatrix, y *matrix.DenseMatrix) {
	rows, cols := x.Rows(), x.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			x.Set(i, j, x.Get(i, j)-y.Get(0, j))
		}
	}
}

func mean(x *matrix.DenseMatrix) *matrix.DenseMatrix {
	rows, cols := x.Rows(), x.Cols()
	m, n := matrix.Zeros(1, cols), 0
	for i := 0; i < rows; i++ {
		zero := true
		for j := 0; j < cols; j++ {
			xij := x.Get(i, j)
			if xij != 0 {
				zero = false
			}
			m.Set(0, j, m.Get(0, j)+xij)
		}
		/* zero vectors skew the mean preventing good centering*/
		if !zero {
			n++
		}
	}
	for j := 0; j < cols; j++ {
		m.Set(0, j, m.Get(0, j)/float64(n))
	}
	return m
}

func cov(x *matrix.DenseMatrix) *matrix.DenseMatrix {
	y := x.Transpose()
	z, err := y.TimesDense(x)
	if err != nil {
		log.Fatal(err)
	}
	N, rows, cols := float64(x.Rows()), z.Rows(), z.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			z.Set(i, j, z.Get(i, j)/N)
		}
	}
	return z
}

/* http://nghiaho.com/?page_id=1030 */
func PCA(m *matrix.DenseMatrix, size int) *matrix.DenseMatrix {
	//norm(m)
	subtract(m, mean(m))
	var v *matrix.DenseMatrix
	{
		_, _, V, err := cov(m).SVD()
		if err != nil {
			log.Fatal(err)
		}
		rows := V.Rows()
		v = matrix.Zeros(rows, size)
		for i := 0; i < rows; i++ {
			for j := 0; j < size; j++ {
				v.Set(i, j, V.Get(i, j))
			}
		}
	}
	r, err := m.TimesDense(v)
	if err != nil {
		log.Fatal(err)
	}
	return r
}

type CPoint struct {
	X, Y float64
	C    int
	R, B byte
}

type CPoints struct {
	CPoints []CPoint
	line    bool
}

func (cp *CPoints) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, trY := plt.Transforms(&c)
	var lastX, lastY vg.Length
	for i, point := range cp.CPoints {
		x := trX(point.X)
		y := trY(point.Y)

		var p vg.Path
		p.Move(x+1, y)
		p.Arc(x, y, 1, 0, 2*math.Pi)
		p.Close()
		if point.C < 0 {
			c.SetColor(color.RGBA{0, point.R, point.B, 255})
		} else {
			switch point.C {
			case 0:
				c.SetColor(color.RGBA{0, 255, 0, 255})
			case 1:
				c.SetColor(color.RGBA{255, 0, 0, 255})
			case 2:
				c.SetColor(color.RGBA{0, 0, 255, 255})
			}
		}
		c.Fill(p)
		if cp.line && i > 0 {
			var p vg.Path
			p.Move(lastX, lastY)
			p.Line(x, y)
			c.Stroke(p)
		}
		lastX, lastY = x, y
	}
}

func (cp *CPoints) DataRange() (xmin, xmax, ymin, ymax float64) {
	xmin, xmax, ymin, ymax = math.MaxFloat64, 0, math.MaxFloat64, 0
	for _, point := range cp.CPoints {
		if point.X < xmin {
			xmin = point.X
		}
		if point.X > xmax {
			xmax = point.X
		}
		if point.Y < ymin {
			ymin = point.Y
		}
		if point.Y > ymax {
			ymax = point.Y
		}
	}
	return
}

func yearIterator(prices []Quote, samples int, norm bool, iterate func(year int, patterns [][][]float64)) {
	patterns, i, size, min, max := [][][]float64{}, 0, len(prices), math.MaxFloat64, float64(0)
	normalize := func() {}
	if norm {
		normalize = func() {
			diff := max - min
			for _, pattern := range patterns {
				for _, a := range pattern {
					for b, _ := range a {
						a[b] = (a[b] - min) / diff
					}
				}
			}
		}
	}
	last := samples - 1
	for i+last*7 < size {
		if prices[i].t.Year() != prices[i+last*7].t.Year() {
			normalize()
			iterate(prices[i].t.Year(), patterns)
			patterns, min, max = [][][]float64{}, math.MaxFloat64, float64(0)
			i += last * 7
			if i+last*7 >= size {
				break
			}
		}
		in, out := make([]float64, samples), make([]float64, samples)
		for j := 0; j < samples; j++ {
			price := prices[i+j*7].price
			if price < min {
				min = price
			}
			if price > max {
				max = price
			}
			in[j] = price
			out[j] = price
		}
		patterns = append(patterns, [][]float64{in, out})
		i++
	}
	if len(patterns) > 0 {
		normalize()
		iterate(prices[i-1].t.Year(), patterns)
	}
}

func rnnVectorizer(prices []Quote, minYear, maxYear int) *matrix.DenseMatrix {
	iterate := func(year int, patterns [][][]float64) {
		maxYear = year
	}
	yearIterator(prices, 4, false, iterate)

	model := matrix.Zeros(maxYear-minYear+1, MODEL_WIDTH)
	iterate = func(year int, patterns [][][]float64) {
		fmt.Println(year)
		rand.Seed(1)
		ff := &gobrain.FeedForward{}
		ff.Init(NEURAL_WIDTH, NEURAL_MIDDLE, NEURAL_WIDTH)
		ff.SetContexts(7, nil)
		ff.Train(patterns, 1, 0.6, 0.4, true)
		i, j := year-minYear, 0
		for _, weights := range ff.InputWeights {
			for _, weight := range weights {
				model.Set(i, j, weight)
				j++
			}
		}
		for _, weights := range ff.OutputWeights {
			for _, weight := range weights {
				model.Set(i, j, weight)
				j++
			}
		}
	}
	yearIterator(prices, 4, true, iterate)

	return model
}

func rnniVectorizer(prices []Quote, minYear, maxYear int) *matrix.DenseMatrix {
	iterate := func(year int, patterns [][][]float64) {
		maxYear = year
	}
	yearIterator(prices, 4, false, iterate)

	model := matrix.Zeros(maxYear-minYear+1, maxYear-minYear+1)
	iterate = func(year int, patterns [][][]float64) {
		fmt.Println(year)
		rand.Seed(1)
		ff := &gobrain.FeedForward{}
		ff.Init(NEURAL_WIDTH, NEURAL_MIDDLE, NEURAL_WIDTH)
		ff.SetContexts(7, nil)
		ff.Train(patterns, 1, 0.6, 0.4, true)
		iterate := func(y int, patterns [][][]float64) {
			err := 0.0
			ff.SetContexts(7, nil)
			for _, pattern := range patterns {
				out := ff.Update(pattern[0])
				for a, b := range out {
					c := b - pattern[0][a]
					err += c * c
				}
			}
			model.Set(year-minYear, y-minYear, err/float64(len(patterns)))
		}
		yearIterator(prices, 4, true, iterate)
	}
	yearIterator(prices, 4, true, iterate)

	return model
}

func posNorm(_, _ int, _ float64) float64 {
	return math.Abs(rand.NormFloat64())
}

func nmfVectorizer(prices []Quote, minYear, maxYear int) *matrix.DenseMatrix {
	iterate := func(year int, patterns [][][]float64) {
		maxYear = year
	}
	yearIterator(prices, SAMPLES, false, iterate)

	model := matrix.Zeros(maxYear-minYear+1, SAMPLES*NMF_WIDTH)
	conf := nmf.Config{
		Tolerance:   1e-3,
		MaxIter:     100,
		MaxOuterSub: 1000,
		MaxInnerSub: 20,
		Limit:       30 * time.Minute,
	}
	iterate = func(year int, patterns [][][]float64) {
		fmt.Println(year)
		data := []float64{}
		for _, pattern := range patterns {
			data = append(data, pattern[0]...)
		}

		cols, rows := SAMPLES, len(data)/SAMPLES
		rand.Seed(1)
		V := mat64.NewDense(rows, cols, data)
		Wo := mat64.NewDense(rows, NMF_WIDTH, nil)
		Wo.Apply(posNorm, Wo)
		Ho := mat64.NewDense(NMF_WIDTH, cols, nil)
		Ho.Apply(posNorm, Ho)
		W, H, ok := nmf.Factors(V, Wo, Ho, conf)
		if !ok {
			fmt.Println("matrix factoring failed")
		}
		_ = W
		k := 0
		for i := 0; i < cols; i++ {
			for j := 0; j < NMF_WIDTH; j++ {
				model.Set(year-minYear, k, H.At(j, i))
				k++
			}
		}
	}
	yearIterator(prices, SAMPLES, false, iterate)

	return model
}

var vectorizers = []Vectorizer{
	{"nmf", "non-negative matrix factorization", nmfVectorizer},
	{"rnn", "recurrent neural network", rnnVectorizer},
	{"rnni", "recurrent neural network indirect", rnniVectorizer},
}

func (v *Vectorizer) Graph(prices []Quote, minYear, maxYear int) {
	model := v.Vectorize(prices, minYear, maxYear)
	//norm(model)
	rows, cols := model.Rows(), model.Cols()
	rawData := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		rawData[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			rawData[i][j] = model.Get(i, j)
		}
	}
	labels, _, err := kmeans.Kmeans(rawData, 3, kmeans.CanberraDistance, 10)
	if err != nil {
		log.Fatal(err)
	}

	r := PCA(model.Copy(), 2)
	rows = r.Rows()
	points := make([]CPoint, rows)
	for i := 0; i < rows; i++ {
		points[i].X, points[i].Y, points[i].C =
			r.Get(i, 0), r.Get(i, 1), labels[i]
	}
	marketDynamics := &CPoints{points, false}

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = v.Name + " Market Dynamics"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(marketDynamics)
	if err := p.Save(512, 512, v.Abbr+"_market_dynamics.png"); err != nil {
		log.Fatal(err)
	}

	ticks := []plot.Tick{}
	for _, price := range prices {
		if price.t.Year()%2 == 0 && price.t.YearDay() == 1 {
			ticks = append(ticks, plot.Tick{float64(price.t.Unix()), fmt.Sprintf("%v", price.t.Year())})
		}
	}

	cpoints := make([]CPoint, len(prices))
	for i, price := range prices {
		cpoints[i].X, cpoints[i].Y, cpoints[i].C =
			float64(price.t.Unix()), price.price, -1 //labels[price.t.Year()-minYear]
	}
	marketPrices := &CPoints{cpoints, false}
	xmin, xmax, ymin, ymax := marketDynamics.DataRange()
	for i, price := range prices {
		index := price.t.Year() - minYear
		if index < len(points) {
			point := points[index]
			cpoints[i].R = byte(255 * (point.X - xmin) / (xmax - xmin))
			cpoints[i].B = byte(255 * (point.Y - ymin) / (ymax - ymin))
		} else {
			cpoints[i].R = 255
		}
	}

	p, err = plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = v.Name + " DJIA"
	p.X.Label.Text = "Time"
	p.Y.Label.Text = "Price"
	p.X.Tick.Marker = plot.ConstantTicks(ticks)
	p.Add(marketPrices)
	if err := p.Save(2048, 2048, v.Abbr+"_market_prices.png"); err != nil {
		log.Fatal(err)
	}

	var markov [3][3]int
	last := 0
	for _, label := range labels {
		markov[last][label]++
		last = label
	}
	colors := [...]string{"Green", "Red", "Blue"}
	for j, a := range markov {
		sum := 0.0
		for _, b := range a {
			sum += float64(b)
		}
		fmt.Println(colors[j])
		for i, b := range a {
			fmt.Printf("%.1f %v\n", 100*float64(b)/sum, colors[i])
		}
		fmt.Printf("\n")
	}

	rows, cols = model.Rows(), model.Cols()
	angles := make([]float64, rows)
	for i := 1; i < rows; i++ {
		a, b := make(Vector, cols), make(Vector, cols)
		for j := 0; j < cols; j++ {
			a[j], b[j] = model.Get(i, j), model.Get(i-1, j)
		}
		angles[i] = a.similarity(b)
	}

	points = make([]CPoint, rows)
	for i := 0; i < rows; i++ {
		unix := time.Date(minYear+i, time.January, 0, 0, 0, 0, 0, time.UTC).Unix()
		points[i].X, points[i].Y, points[i].C =
			float64(unix), angles[i], labels[i]
	}
	marketSimilarity := &CPoints{points, true}

	p, err = plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Title.Text = v.Name + " Market Similarity"
	p.X.Label.Text = "Time"
	p.Y.Label.Text = "Similarity"
	p.X.Tick.Marker = plot.ConstantTicks(ticks)
	p.Add(marketSimilarity)
	if err := p.Save(2048, 2048, v.Abbr+"_market_similarity.png"); err != nil {
		log.Fatal(err)
	}
}

var vectorizer = flag.String("vectorizer", "all", "all/nmf/rnn/rnni")

func main() {
	flag.Parse()

	in, err := os.Open("UDJIAD1.csv")
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(in)
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	data = data[1:]
	minYear, maxYear, size := time.Now().Year()+1, 0, 0
	iterator(data, func(index int, t time.Time, price float64) {
		size++
		year := t.Year()
		if year < minYear {
			minYear = year
		}
		if year > maxYear {
			maxYear = year
		}
		//fmt.Printf("%v %v\n", t, price)
	})
	prices := make([]Quote, size)
	iterator(data, func(index int, t time.Time, price float64) {
		i := size - 1 - index
		prices[i].t = t
		prices[i].price = price
	})

	if *vectorizer == "all" {
		for _, v := range vectorizers {
			v.Graph(prices, minYear, maxYear)
		}
	} else {
		for _, v := range vectorizers {
			if v.Abbr == *vectorizer {
				v.Graph(prices, minYear, maxYear)
				break
			}
		}
	}
}
