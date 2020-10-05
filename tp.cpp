// -------------------------------------------
// gMini : a minimal OpenGL/GLUT application
// for 3D graphics.
// Copyright (C) 2006-2008 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

// -------------------------------------------
// Disclaimer: this code is dirty in the
// meaning that there is no attention paid to
// proper class attribute access, memory
// management or optimisation of any kind. It
// is designed for quick-and-dirty testing
// purpose.
// -------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include "src/jmkdtree.h"


BasicANNkdTree kdtree;

std::vector< Vec3 > positions;
std::vector< Vec3 > normals;

std::vector< Vec3 > positions2; // Output final, après HPSS et bruit
std::vector< Vec3 > output_fonction; // Output HPSS
std::vector< Vec3 > normals2;

float noise = 0.0;

bool original = true;
bool normals_disp = false;
float kernel_radius = 1.0;

u_int nb_pts_proj = 20000;

// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 640;
static unsigned int SCREENHEIGHT = 480;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;




// ------------------------------------------------------------------------------------------------------------
// i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN (const std::string & filename , std::vector< Vec3 > & o_positions , std::vector< Vec3 > & o_normals ) {
    unsigned int surfelSize = 6;
    FILE * in = fopen (filename.c_str (), "rb");
    if (in == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    size_t READ_BUFFER_SIZE = 1000; // for example...
    float * pn = new float[surfelSize*READ_BUFFER_SIZE];
    o_positions.clear ();
    o_normals.clear ();
    while (!feof (in)) {
        unsigned numOfPoints = fread (pn, 4, surfelSize*READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) {
            o_positions.push_back (Vec3 (pn[i], pn[i+1], pn[i+2]));
            o_normals.push_back (Vec3 (pn[i+3], pn[i+4], pn[i+5]));
        }

        if (numOfPoints < surfelSize*READ_BUFFER_SIZE) break;
    }
    fclose (in);
    delete [] pn;
}



void savePN (const std::string & filename , std::vector< Vec3 > const & o_positions , std::vector< Vec3 > const & o_normals ) {
    if ( o_positions.size() != o_normals.size() ) {
        std::cout << "The pointset you are trying to save does not contain the same number of points and normals." << std::endl;
        return;
    }
    FILE * outfile = fopen (filename.c_str (), "wb");
    if (outfile == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    for(unsigned int pIt = 0 ; pIt < o_positions.size() ; ++pIt) {
        fwrite (&(o_positions[pIt]) , sizeof(float), 3, outfile);
        fwrite (&(o_normals[pIt]) , sizeof(float), 3, outfile);
    }
    fclose (outfile);
}



void scaleAndCenter( std::vector< Vec3 > & io_positions ) {
    Vec3 bboxMin( FLT_MAX , FLT_MAX , FLT_MAX );
    Vec3 bboxMax( FLT_MIN , FLT_MIN , FLT_MIN );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        for( unsigned int coord = 0 ; coord < 3 ; ++coord ) {
            bboxMin[coord] = std::min<float>( bboxMin[coord] , io_positions[pIt][coord] );
            bboxMax[coord] = std::max<float>( bboxMax[coord] , io_positions[pIt][coord] );
        }
    }
    Vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
    float bboxLongestAxis = std::max<float>( bboxMax[0]-bboxMin[0] , std::max<float>( bboxMax[1]-bboxMin[1] , bboxMax[2]-bboxMin[2] ) );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = (io_positions[pIt] - bboxCenter) / bboxLongestAxis;
    }
}



void applyRandomRigidTransformation( std::vector< Vec3 > & io_positions , std::vector< Vec3 > & io_normals ) {
    srand(time(NULL));
    Mat3 R = Mat3::RandRotation();
    Vec3 t = Vec3::Rand(1.f);
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = R * io_positions[pIt] + t;
        io_normals[pIt] = R * io_normals[pIt];
    }
}


void subsample( std::vector< Vec3 > & i_positions , std::vector< Vec3 > & i_normals , float minimumAmount = 0.1f , float maximumAmount = 0.2f ) {
    std::vector< Vec3 > newPos , newNormals;
    std::vector< unsigned int > indices(i_positions.size());
    for( unsigned int i = 0 ; i < indices.size() ; ++i ) indices[i] = i;
    srand(time(NULL));
    std::random_shuffle(indices.begin() , indices.end());
    unsigned int newSize = indices.size() * (minimumAmount + (maximumAmount-minimumAmount)*(float)(rand()) / (float)(RAND_MAX));
    newPos.resize( newSize );
    newNormals.resize( newSize );
    for( unsigned int i = 0 ; i < newPos.size() ; ++i ) {
        newPos[i] = i_positions[ indices[i] ];
        newNormals[i] = i_normals[ indices[i] ];
    }
    i_positions = newPos;
    i_normals = newNormals;
}

bool save( const std::string & filename , std::vector< Vec3 > & vertices , std::vector< unsigned int > & triangles ) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    }

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = vertices.size() , n_triangles = triangles.size()/3;
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        myfile << vertices[v][0] << " " << vertices[v][1] << " " << vertices[v][2] << std::endl;
    }
    for( unsigned int f = 0 ; f < n_triangles ; ++f ) {
        myfile << 3 << " " << triangles[3*f] << " " << triangles[3*f+1] << " " << triangles[3*f+2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}



// ------------------------------------------------------------------------------------------------------------
// rendering.
// ------------------------------------------------------------------------------------------------------------

void initLight () {
    GLfloat light_position1[4] = {22.0f, 16.0f, 50.0f, 0.0f};
    GLfloat direction1[3] = {-52.0f,-16.0f,-50.0f};
    GLfloat color1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat ambient[4] = {0.3f, 0.3f, 0.3f, 0.5f};

    glLightfv (GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv (GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv (GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv (GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv (GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable (GL_LIGHT1);
    glEnable (GL_LIGHTING);
}

void init () {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    initLight ();
    glCullFace (GL_BACK);
    glEnable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
}



void drawTriangleMesh( std::vector< Vec3 > const & i_positions , std::vector< unsigned int > const & i_triangles ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_triangles.size() / 3 ; ++tIt) {
        Vec3 p0 = i_positions[3*tIt];
        Vec3 p1 = i_positions[3*tIt+1];
        Vec3 p2 = i_positions[3*tIt+2];
        Vec3 n = Vec3::cross(p1-p0 , p2-p0);
        n.normalize();
        glNormal3f( n[0] , n[1] , n[2] );
        glVertex3f( p0[0] , p0[1] , p0[2] );
        glVertex3f( p1[0] , p1[1] , p1[2] );
        glVertex3f( p2[0] , p2[1] , p2[2] );
    }
    glEnd();
}

void drawPointSet( std::vector< Vec3 > const & i_positions , std::vector< Vec3 > const & i_normals, bool norm_colo = false ) {
    glBegin(GL_POINTS);
    for(unsigned int pIt = 0 ; pIt < i_positions.size() ; ++pIt) {
        if(normals_disp && norm_colo){glColor3f(abs(i_normals[pIt][0]) , abs(i_normals[pIt][1]) , abs(i_normals[pIt][2]));}
        glNormal3f( i_normals[pIt][0] , i_normals[pIt][1] , i_normals[pIt][2] );
        glVertex3f( i_positions[pIt][0] , i_positions[pIt][1] , i_positions[pIt][2] );
    }
    glEnd();
}


void draw () {

    if(original){
        glPointSize(2); // for example...

        glColor3f(0.8, 0.8, 1.0);
        drawPointSet(positions , normals);
    }

    glPointSize(3); // for example...

    glColor3f(1.0, 0.2, 0.2);
    drawPointSet(positions2 , normals2, true);
}








void display () {
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    draw ();
    glFlush ();
    glutSwapBuffers ();
}

void idle () {
    glutPostRedisplay ();
}

void init_points_set(double taille_cote, u_int size_pts_set, double shrink=1.0);
void launch_hpss(const BasicANNkdTree& kdtree, float k_size = 1.0);
void noisify();

void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
        case 'f':
            if (fullScreen == true) {
                glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
                fullScreen = false;
            } else {
                glutFullScreen ();
                fullScreen = true;
            }
            break;

        case 'w':
            GLint polygonMode[2];
            glGetIntegerv(GL_POLYGON_MODE, polygonMode);
            if(polygonMode[0] != GL_FILL)
                glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
            else
                glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
            break;

        case 'o':
            original = !original;
            break;

        case 'n':
            normals_disp = !normals_disp;
            break;

        case 'p':
            kernel_radius += 0.05;
            init_points_set(2.0, nb_pts_proj, 0.8);
            launch_hpss(kdtree, kernel_radius);
            noisify();
            break;

        case 'm':
            kernel_radius -= 0.05;
            init_points_set(2.0, nb_pts_proj, 0.8);
            launch_hpss(kdtree, kernel_radius);
            noisify();
            break;

        case 'i':
            nb_pts_proj += 10;
            init_points_set(2.0, nb_pts_proj, 0.8);
            launch_hpss(kdtree, kernel_radius);
            noisify();
            break;

        case 'k':
            nb_pts_proj -= 10;
            init_points_set(2.0, nb_pts_proj, 0.8);
            launch_hpss(kdtree, kernel_radius);
            noisify();
            break;

        case 'u':
            noise += 0.005;
            noisify();
            break;

        case 'j':
            noise = (noise >= 0.005) ? (noise - 0.005) : (0.0);
            noisify();
            break;


        default:
            break;
    }
    idle ();
}

void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize (w, h);
}

// Si ne compile pas: forcer la compilation en -std=c++11 (ou ulterieur) ou remplacer les auto par des double

auto max_array(const auto* tab, u_int n){
    auto max = tab[0];
    for(u_int i = 0 ; i < n ; i++){
        if(tab[i] > max){
            max = tab[i];
        }
    }
    return max;
}

auto sum_array(const auto* tab, u_int n){
    auto accumulateur = 0.0;

    for(u_int i = 0 ; i < n ; i++){
        accumulateur += tab[i];
    }

    return accumulateur;
}

// Initialise l'ensemble de points artificiels qui sera projeté
// taille_cote = N  ==>  Chaque point est dans le cube dont les extremums sur chaque axes sont -N et N
// shrink permet de rapprocher le nouveau nuage du centre du monde (si 0 < shrink <= 1)

void init_points_set(double taille_cote, u_int size_pts_set, double shrink){
    positions2.clear();
    output_fonction.clear();
    positions2.resize(size_pts_set);
    normals2.resize(positions2.size());

    for( unsigned int pIt = 0 ; pIt < positions2.size() ; pIt++ ) {
        positions2[pIt] = Vec3(
                        ((double)(rand())/(double)(RAND_MAX) * 2.0 * taille_cote) - taille_cote,
                        ((double)(rand())/(double)(RAND_MAX) * 2.0 * taille_cote) - taille_cote,
                        ((double)(rand())/(double)(RAND_MAX) * 2.0 * taille_cote) - taille_cote
                    );
        positions2[pIt].normalize();
        positions2[pIt] *= shrink;
    }
}

// 0: Uniforme
// 1: Gaussien
// 2: Wendland
// 3: Singulier

void process_weights(double** weights, const ANNidxArray id_nearest_neighbors, const ANNdistArray square_distances_to_neighbors, uint8_t k_type, u_int knn, float radius = 1.0){
    *weights = new double [knn];

    // Uniforme, sans calcul
    if(k_type == 0){
        for(u_int i = 0 ; i < knn ; i++){
            (*weights)[i] = 1.0 / (double)knn;
        }
    }

    // Gaussien
    else if(k_type == 1){
        float dist_max = max_array(square_distances_to_neighbors, knn);

        for(u_int i = 0 ; i < knn ; i++){
            if(square_distances_to_neighbors[i] > radius){ // Le point est trop loin
                (*weights)[i] = 0.0;
            }
            else{
                (*weights)[i] = exp((-square_distances_to_neighbors[i]) / dist_max);
            }
        }
    }
}


void projection(const Vec3& point, Vec3& p_point, u_int idx_neighbor, const std::vector<Vec3>& positions, const std::vector<Vec3>& normals, double s = 0.5){
    p_point = point - (Vec3::dot(point - positions[idx_neighbor], normals[idx_neighbor])) * normals[idx_neighbor];
    p_point = (s * p_point) + ((1.0 - s) * p_point);
}


u_int centroid_and_normal(const double* poids, const std::vector< Vec3 >& projected_points, const std::vector<Vec3>& normals, u_int knn, Vec3& centroid, Vec3& normal, const ANNidxArray id_nearest_neighbors){
    centroid = Vec3(0.0, 0.0, 0.0);
    normal = Vec3(0.0, 0.0, 0.0);
    double diviseur = sum_array(poids, knn);

    if(diviseur != 0.0){
        for(u_int i = 0 ; i < knn ; i++){
            centroid += (poids[i] * projected_points[i]);
            normal += (poids[i] * normals[id_nearest_neighbors[i]]);
        }

        centroid /= diviseur;
        normal /= diviseur;

        return 0;
    }
    return 1;
}


u_int HPSS(Vec3 inputPoint, Vec3& outputPoint, Vec3& outputNormal, const std::vector<Vec3>& positions, const std::vector<Vec3>& normals, const BasicANNkdTree& kdtree, uint8_t kernel_type, float radius, u_int nb_iterations=10, u_int knn=20){
    Vec3 x_k = inputPoint;

    for(u_int i = 0 ; i < nb_iterations ; i++){

        ANNidxArray id_nearest_neighbors = new ANNidx[knn];
        ANNdistArray square_distances_to_neighbors = new ANNdist[knn];
        std::vector< Vec3 > projected_points;
        double* poids = NULL;

        // 1. Récupération des K-nearest-neighbors:
        kdtree.knearest(x_k, knn, id_nearest_neighbors, square_distances_to_neighbors);

        // 2. Calcul des poids en fonction du type
        process_weights(&poids, id_nearest_neighbors, square_distances_to_neighbors, kernel_type, knn, radius);

        // 3. Calcul du projeté du point d'entrée sur chacun des plans déduits à partir des normales des voisins
        for(u_int j = 0 ; j < knn ; j++){
            Vec3 projected_point;
            projection(x_k, projected_point, id_nearest_neighbors[j], positions, normals);
            projected_points.push_back(projected_point);
        }

        // 4. Calcul du centroide et de la nouvelle normale
        Vec3 centroid, normal;
        u_int res = centroid_and_normal(poids, projected_points, normals, knn, centroid, normal, id_nearest_neighbors);
        if(res){return 1;} // Ne passe ici qu'en cas de division par 0 avec les poids

        // 5. Refresh des valeurs avant nouvelle itération
        outputPoint = centroid;
        x_k = centroid;
        outputNormal = normal;

        delete [] poids;
        delete [] id_nearest_neighbors;
        delete [] square_distances_to_neighbors;
    }
    return 0;
}

void noisify(){
    positions2 = output_fonction;

    for(u_int i = 0 ; i < positions2.size() ; i++){
        float magnitude = (float)(rand())/(float)(RAND_MAX);
        magnitude *= (2.0 * noise);
        magnitude -= noise;
        positions2[i] += (magnitude * normals2[i]);
    }
}

void launch_hpss(const BasicANNkdTree& kdtree, float k_size){
    u_int borne = positions2.size();

    for(u_int i = 0 ; i < borne ; i++){
        Vec3 point, normal;
        u_int success = HPSS(positions2[0], point, normal, positions, normals, kdtree, 1, k_size);
        positions2.erase(positions2.begin()+0);
        normals2.erase(normals2.begin()+0);

        if(!success){
            output_fonction.push_back(point);
            //positions2.push_back(point);
            normals2.push_back(normal);
        }
    }
    std::cout << borne << " points projetés.  |  " << output_fonction.size() << " correctements projetés  |  Kernel_size = " << k_size << std::endl;
}

int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("TP - Points processing");

    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);
    std::cout << "[O] : Afficher/masquer points d'origines" << std::endl;
    std::cout << "[N] : Afficher/masquer couleurs d'après normales" << std::endl;
    std::cout << "[M/P] : Diminuer/Augmenter la taille du kernel" << std::endl;
    std::cout << "[K/I] : Diminuer/Augmenter le nombre de points projetés" << std::endl;
    std::cout << "[J/U] : Diminuer/Augmenter le bruit le long des normales" << std::endl;

    {
        // Load a first pointset, and build a kd-tree:
        loadPN("pointsets/igea.pn" , positions , normals);
        kdtree.build(positions);

        // Create a second pointset that is artificial, and project it on pointset1 using MLS techniques:
        init_points_set(2.0, nb_pts_proj, 0.8);
        launch_hpss(kdtree, kernel_radius);
        noisify();
    }



    glutMainLoop ();
    return EXIT_SUCCESS;
}
