using System;
using Ezc3d;

// This example shows how to use ezc3d in C#. It reads a c3d file and prints some information about it.
// It therefore assumes that the c3d files were extracted
// To run the example, compile the ezc3d project with example turned on and the *.csproj will be generated 
// in the current folder. You can then run the example with "dotnet run" in the terminal from the current folder.

class Ezc3dExample {
    static void Main() {
        var file = new c3d("../c3dFiles/ezc3d-testFiles-master/ezc3d-testFiles-master/Vicon.c3d");
        
        // Showcasing how to get header information
        Console.WriteLine("Header information:");
        Console.WriteLine("- Number of points: " + file.header().nb3dPoints());

        // Showcasing how to get parameters
        Console.WriteLine("Available parameters:");
        for (uint i = 0; i < file.parameters().nbGroups(); i++) {
            Console.WriteLine("- " + file.parameters().group(i).name());
        }

        var used = file.parameters().group("POINT").parameter("USED").valuesAsInt();
        Console.WriteLine("Number of points: " + used[0]);

        // Showcasing how to get data
        Console.WriteLine("First point of the first frame:");
        var point = file.data().frame(0).points().point(0);
        Console.WriteLine("- X: " + point.x());
        Console.WriteLine("- Y: " + point.y());
        Console.WriteLine("- Z: " + point.z());
    }
}
